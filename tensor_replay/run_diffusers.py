"""ComfyUI 조건 이후의 원본 Diffusers 모듈형 블록 실행."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, parse_run, add_source, Recorder, read_json
from inputs import load_inputs, finish_image


def build_blocks():
    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
    from diffusers.modular_pipelines.anima.before_denoise import (
        AnimaTextInputStep, AnimaPrepareLatentsStep, AnimaSetTimestepsStep)
    from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseStep
    from diffusers.modular_pipelines.anima.modular_blocks_anima import AnimaDecodeStep

    class DownstreamBlocks(SequentialPipelineBlocks):
        model_name = "anima"
        block_classes = [AnimaTextInputStep, AnimaPrepareLatentsStep, AnimaSetTimestepsStep,
                         AnimaDenoiseStep, AnimaDecodeStep]
        block_names = ["input", "prepare_latents", "set_timesteps", "denoise", "decode"]
    return DownstreamBlocks()


def run(args, record):
    import torch
    from diffusers import (CosmosTransformer3DModel, AutoencoderKLQwenImage,
                           FlowMatchEulerDiscreteScheduler, ClassifierFreeGuidance, ComponentsManager)
    sys.path.insert(0, str(ROOT / "diffusers"))
    from fp16_debug.residual_fp32 import install_residual_fp32
    context, noise, sigmas = load_inputs(args, record)
    dtype = getattr(torch, args.options["dtype"])
    converted = args.models / "diffusers_converted" / args.options["dtype"]
    manifest = read_json(args.models / "model_manifest.json")
    expected = dict(weights=manifest["weights"], dtype=args.options["dtype"],
                    source_commit=read_json(ROOT / "source_manifest.json")["diffusers"]["commit"])
    with record.stage("convert_models"):
        for component in ("transformer", "vae"):
            marker = converted / component / "conversion_source.json"
            if not marker.exists() or read_json(marker) != expected:
                subprocess.run([sys.executable, str(ROOT / "diffusers/convert_weights.py"),
                    "--models", str(args.models), "--component", component, "--dtype", args.options["dtype"]], check=True)
    with record.stage("load_models"):
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda:0")
        pipe = build_blocks().init_pipeline(components_manager=manager)
        pipe.update_components(
            transformer=CosmosTransformer3DModel.from_pretrained(converted / "transformer", torch_dtype=dtype, local_files_only=True),
            vae=AutoencoderKLQwenImage.from_pretrained(converted / "vae", torch_dtype=dtype, local_files_only=True),
            scheduler=FlowMatchEulerDiscreteScheduler(shift=1.0),
            guider=ClassifierFreeGuidance(guidance_scale=args.hp["cfg"]))
        record.handles.extend(install_residual_fp32(pipe.transformer))
        record.observe_model("dit", pipe.transformer)
    record.data.update(effective_sampler="native Diffusers FlowMatchEulerDiscreteScheduler",
                       residual_fp32=True, computation_replacements=True,
                       scheduler_shift=1.0, scheduler_note="input sigmas already shifted by ComfyUI")
    original_step = pipe.scheduler.step
    original_set = pipe.scheduler.set_timesteps
    original_decode = pipe.vae.decode
    state = {"i": 0}
    from functools import wraps
    @wraps(original_set)
    def set_timesteps(*a, **kw):
        result = original_set(*a, **kw)
        if not torch.equal(pipe.scheduler.sigmas.cpu(), sigmas):
            raise ValueError("주입한 ComfyUI sigma 시간표가 변경되었습니다.")
        return result
    @wraps(original_step)
    def step(model_output, timestep, sample, *a, **kw):
        i = state["i"]
        record.save("x_before", sample, i)
        record.save("velocity", model_output, i)
        record.save("denoised", sample.float()-sigmas[i].to(sample.device)*model_output.float(), i)
        result = original_step(model_output, timestep, sample, *a, **kw)
        value = result[0] if isinstance(result, tuple) else result.prev_sample
        record.save("x_after", value, i)
        state["i"] += 1
        return result
    @wraps(original_decode)
    def decode(z, *a, **kw):
        record.save("final_vae_latent", z)
        result = original_decode(z, *a, **kw)
        record.save("decoded_pixels", result[0] if isinstance(result, tuple) else result.sample)
        return result
    pipe.scheduler.set_timesteps, pipe.scheduler.step, pipe.vae.decode = set_timesteps, step, decode
    try:
        with record.stage("pipeline_total"):
            result = pipe(prompt_embeds=context[1:2].to("cuda"), negative_prompt_embeds=context[0:1].to("cuda"),
                          latents=noise.to("cuda"), height=args.hp["height"], width=args.hp["width"],
                          num_inference_steps=args.hp["steps"], sigmas=sigmas[:-1].tolist(), output=["images", "latents"])
    finally:
        pipe.scheduler.set_timesteps, pipe.scheduler.step, pipe.vae.decode = original_set, original_step, original_decode
    record.save("final_model_latent", result["latents"])
    finish_image(result["images"][0], result["latents"], args, record)


if __name__ == "__main__":
    args = parse_run("diffusers")
    add_source("diffusers")
    record = Recorder(args)
    try:
        run(args, record)
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish()
