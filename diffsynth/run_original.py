"""원본 DiffSynth Anima를 실행하고 공통 형식의 비교 기록을 저장한다."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, Recorder, add_source, parse_run, read_json, write_json
from trace_original import trace_pipeline


def run(args, record):
    import torch
    from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
    from diffsynth.core.attention.attention import ATTENTION_IMPLEMENTATION

    hp, options = args.hp, args.options
    dtype = getattr(torch, options["dtype"])
    manifest = read_json(args.models / "model_manifest.json")
    tokenizers = manifest["tokenizer_sources"]
    vram = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype, onload_device="cpu",
                preparing_dtype=dtype, preparing_device="cuda", computation_dtype=dtype, computation_device="cuda")
    record.data.update(mode="original", effective_sampler="FlowMatchScheduler Euler",
                       effective_schedule="original Z-Image grid", precision_policy=options["dtype"],
                       attention_dispatch=ATTENTION_IMPLEMENTATION, computation_replacements=False,
                       trace_notes={"latent_axes": "B,C,T,H,W; saved view only",
                                    "denoised": "FP32 diagnostic x-sigma*velocity, never passed to solver",
                                    "pixels": "actual uint8 image normalized to 0..1",
                                    "final_vae_latent": "actual conv2 input after native denormalization"})
    with record.stage("load_models"):
        pipe = AnimaImagePipeline.from_pretrained(
            torch_dtype=dtype, device="cuda",
            model_configs=[ModelConfig(path=str(path), **vram) for path in args.weights.values()],
            tokenizer_config=ModelConfig(path=tokenizers["qwen"]["path"]),
            tokenizer_t5xxl_config=ModelConfig(path=tokenizers["t5"]["path"]),
            vram_limit=torch.cuda.mem_get_info()[0] / 2**30 - options["vram_margin_gib"],
        )
        for name in ("text_encoder", "dit", "vae"):
            record.observe_model(name, getattr(pipe, name))
        record.observe_model("text_adapter", pipe.dit.llm_adapter)
    generation = dict(prompt=hp["prompt"], negative_prompt=hp["negative_prompt"], seed=hp["seed"],
                      height=hp["height"], width=hp["width"], num_inference_steps=hp["steps"],
                      cfg_scale=hp["cfg"], denoising_strength=hp["denoise"],
                      sigma_shift=options["sigma_shift"], rand_device=options["rand_device"])
    write_json(args.output / "effective_generation.json", generation)
    with trace_pipeline(pipe, record), record.stage("pipeline_total"):
        image = pipe(**generation)
    image.save(args.output / "image.png")


if __name__ == "__main__":
    args = parse_run("diffsynth")
    if args.mode != "native":
        raise ValueError("원본 실행은 --mode native만 지원합니다.")
    add_source("diffsynth")
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
