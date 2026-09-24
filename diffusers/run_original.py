"""원본 Diffusers Anima 실행과 관찰용 기록. 계산 함수는 교체하지 않는다."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, Recorder, add_source, parse_run, read_json, write_json
from run import ensure_converted
from trace_original import trace_pipeline


def run(args, record):
    import numpy as np
    import torch
    from diffusers import (AnimaAutoBlocks, AnimaTextConditioner, AutoencoderKLQwenImage,
                           CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler,
                           ComponentsManager, ClassifierFreeGuidance)
    from transformers import Qwen3Model, Qwen2Tokenizer, T5TokenizerFast

    hp, options = args.hp, args.options
    dtype = getattr(torch, options["dtype"])
    args.precision = {name: dtype for name in ("transformer", "text_conditioner", "text_encoder", "vae")}
    record.data.update(mode="original", precision_policy=options["dtype"], computation_replacements=options.get("residual_fp32", True),
                       residual_fp32=options.get("residual_fp32", True), nan_debug=options.get("nan_debug", False),
                       effective_sampler="FlowMatchEulerDiscreteScheduler",
                       effective_schedule="original Anima FlowMatch grid", offload=options["hf_offload"],
                       trace_notes={"latent_axes": "B,C,T,H,W",
                                    "text_adapter_first_output": "native conditioner output including padding",
                                    "denoised": "FP32 diagnostic x-sigma*velocity, never passed to solver",
                                    "pixels": "actual uint8 image normalized to 0..1"})
    with record.stage("convert_models"):
        ensure_converted(args)
    converted = args.models / "diffusers_converted" / options["dtype"]
    tokenizers = read_json(args.models / "model_manifest.json")["tokenizer_sources"]
    with record.stage("load_models"):
        manager = ComponentsManager()
        if options["hf_offload"] == "auto":
            manager.enable_auto_cpu_offload(device="cuda:0")
        elif options["hf_offload"] != "none":
            raise ValueError("hf_offload는 auto 또는 none이어야 합니다.")
        pipe = AnimaAutoBlocks().init_pipeline(components_manager=manager)
        pipe.update_components(
            transformer=CosmosTransformer3DModel.from_pretrained(converted / "transformer", torch_dtype=dtype, local_files_only=True),
            text_conditioner=AnimaTextConditioner.from_pretrained(converted / "text_conditioner", torch_dtype=dtype, local_files_only=True),
            text_encoder=Qwen3Model.from_pretrained(converted / "text_encoder", torch_dtype=dtype, local_files_only=True),
            vae=AutoencoderKLQwenImage.from_pretrained(converted / "vae", torch_dtype=dtype, local_files_only=True),
            tokenizer=Qwen2Tokenizer.from_pretrained(tokenizers["qwen"]["path"], local_files_only=True),
            t5_tokenizer=T5TokenizerFast.from_pretrained(tokenizers["t5"]["path"], local_files_only=True),
            scheduler=FlowMatchEulerDiscreteScheduler(shift=options["shift"]),
            guider=ClassifierFreeGuidance(guidance_scale=hp["cfg"]),
        )
        if options.get("residual_fp32", True):
            from fp16_debug.residual_fp32 import install_residual_fp32
            record.handles.extend(install_residual_fp32(pipe.transformer))
        if options.get("nan_debug", False):
            from fp16_debug.nan_debug import NanDebugger
            debugger = NanDebugger(args.output)
            for name in ("text_encoder", "text_conditioner", "transformer", "vae"):
                # attach마다 새로 추가한 handle만 Recorder에서 정리한다.
                previous = len(debugger.handles)
                debugger.attach(name, getattr(pipe, name))
                record.handles.extend(debugger.handles[previous:])
        if options["hf_offload"] == "none":
            pipe.to("cuda")
        if options["vae_tiling"]:
            pipe.vae.enable_tiling()
        for label, name in (("dit", "transformer"), ("text_adapter", "text_conditioner"),
                            ("text_encoder", "text_encoder"), ("vae", "vae")):
            record.observe_model(label, getattr(pipe, name))
    call = dict(prompt=hp["prompt"], negative_prompt=hp["negative_prompt"], height=hp["height"], width=hp["width"],
                num_inference_steps=hp["steps"], max_sequence_length=options["max_sequence_length"])
    write_json(args.output / "effective_generation.json", dict(call, seed=hp["seed"], cfg=hp["cfg"], shift=options["shift"]))
    outputs = ["images", "qwen_prompt_embeds", "negative_qwen_prompt_embeds", "qwen_attention_mask",
               "negative_qwen_attention_mask", "t5_input_ids", "negative_t5_input_ids", "prompt_embeds", "negative_prompt_embeds"]
    with trace_pipeline(pipe, record), record.stage("pipeline_total"):
        result = pipe(**call, generator=torch.Generator("cpu").manual_seed(hp["seed"]), output=outputs)
    for label, prefix in (("prompt", ""), ("negative_prompt", "negative_")):
        record.save(f"condition_{label}", result[prefix + "qwen_prompt_embeds"])
        record.save(f"condition_{label}_attention_mask", result[prefix + "qwen_attention_mask"])
        ids = result[prefix + "t5_input_ids"]
        record.save(f"condition_{label}_t5xxl_ids", ids.squeeze(0) if ids is not None else None)
        record.save(f"adapted_{label}", result[prefix + "prompt_embeds"])
    image = result["images"][0]
    image.save(args.output / "image.png")
    record.save("pixels", torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255)


if __name__ == "__main__":
    args = parse_run("diffusers")
    if args.mode != "native":
        raise ValueError("원본 실행은 --mode native만 지원합니다.")
    add_source("diffusers")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True)
    (args.output / "project_commit.txt").write_text(commit, encoding="utf-8")
    record = Recorder(args)
    try:
        run(args, record)
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish()
