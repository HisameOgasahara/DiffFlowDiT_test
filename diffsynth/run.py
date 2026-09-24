import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import add_source, parse_run, Recorder, make_noise, make_comfy_sigmas, write_json

def run(args, record):
    import torch
    from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
    hp, options = args.hp, args.options
    if options["max_sequence_length"] != 512:
        raise ValueError("원본 DiffSynth Anima prompt unit의 길이는 512입니다. 길이 변경 실험은 prompt unit에서 명시적으로 수정하세요.")
    dtype = getattr(torch, options["dtype"])
    vram = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype, onload_device="cpu",
                preparing_dtype=dtype, preparing_device="cuda", computation_dtype=dtype, computation_device="cuda")
    with record.stage("load_models"):
        pipe = AnimaImagePipeline.from_pretrained(
            torch_dtype=dtype, device="cuda",
            model_configs=[ModelConfig(path=str(path), **vram) for path in args.weights.values()],
            tokenizer_config=ModelConfig(path=str(args.models / "tokenizers/qwen25_tokenizer")),
            tokenizer_t5xxl_config=ModelConfig(path=str(args.models / "tokenizers/t5_tokenizer")),
            vram_limit=max(0.5, torch.cuda.mem_get_info()[0] / 2**30 - options["diffsynth_vram_margin_gib"]),
        )
        for name in ("dit", "text_encoder", "vae"):
            if getattr(pipe, name) is None:
                raise ValueError(f"원본 DiffSynth 로더가 {name}을 인식하지 못했습니다. 모델 형식과 registry를 확인하세요.")
            record.observe_model(name, getattr(pipe, name))
        record.observe_model("text_adapter", pipe.dit.llm_adapter)
    if options["vae_tiling"]:
        raise ValueError("Anima 원본 pipeline에는 VAE 타일 옵션이 없습니다. 이번 비교에서는 vae_tiling=false를 사용하세요.")
    record.data.update(effective_sampler="FlowMatchScheduler Euler", effective_schedule="Z-Image shifted grid",
                       requested_sampler=hp["sampler_name"], sampling_mismatch_with_png=True,
                       offload=vram, vram_management=pipe.vram_management_enabled)
    from diffsynth.core.attention import attention as attention_module
    record.data["attention_available"] = {k: v for k, v in vars(attention_module).items() if k.endswith("_AVAILABLE") and isinstance(v, bool)}
    original_set = pipe.scheduler.set_timesteps
    def set_timesteps(*pos, **kw):
        original_set(*pos, **kw)
        if args.mode == "matched_euler":
            grid = make_comfy_sigmas(hp["steps"], options["shift"], options["training_steps"])
            pipe.scheduler.sigmas = grid[:-1]
            pipe.scheduler.timesteps = grid[:-1] * options["training_steps"]
            record.data["effective_schedule"] = "ComfyUI simple explicit grid"
        grid = torch.cat([pipe.scheduler.sigmas, torch.zeros(1)])
        record.save("sigmas", grid)
        write_json(args.output / "schedule.json", {"sigmas": grid.tolist(), "model_timesteps": (pipe.scheduler.timesteps / options["training_steps"]).tolist()})
    pipe.scheduler.set_timesteps = set_timesteps
    original_noise = pipe.generate_noise
    def generate_noise(*pos, **kw):
        noise = make_noise(hp).to(pipe.device) if args.mode == "matched_euler" else original_noise(*pos, **kw)
        record.save("initial_noise", noise)
        return noise
    pipe.generate_noise = generate_noise
    original_fn = pipe.model_fn
    def model_fn(**kw):
        # 공통 Euler에서만 solver FP32와 모델 연산 dtype을 분리합니다.
        if args.mode == "matched_euler":
            kw["latents"] = kw["latents"].to(dtype)
        return original_fn(**kw).float() if args.mode == "matched_euler" else original_fn(**kw)
    pipe.model_fn = model_fn
    original_step = pipe.step
    def step(scheduler, latents, progress_id, noise_pred, **kw):
        record.save("x_before", latents, progress_id)
        record.save("velocity", noise_pred, progress_id)
        record.save("denoised", latents.float() - scheduler.sigmas[progress_id].to(latents.device) * noise_pred.float(), progress_id)
        result = original_step(scheduler, latents, progress_id, noise_pred, **kw)
        if progress_id == hp["steps"] - 1:
            record.save("final_model_latent", result)
        return result
    pipe.step = step
    original_unit = pipe.unit_runner
    def unit_runner(unit, pipeline, shared, positive, negative):
        with record.stage(type(unit).__name__):
            shared, positive, negative = original_unit(unit, pipeline, shared, positive, negative)
        for label, values in (("prompt", positive), ("negative_prompt", negative)):
            if "prompt_emb" in values:
                record.save(f"condition_{label}", values["prompt_emb"])
                record.save(f"condition_{label}_t5xxl_ids", values.get("t5xxl_ids"))
        return shared, positive, negative
    pipe.unit_runner = unit_runner
    # 원본 호출을 보존하고 단계별 시간만 측정합니다.
    original_guided = pipe.cfg_guided_model_fn
    def guided(*pos, **kw):
        with record.stage(f"denoise_step_{kw.get('progress_id', 0):03d}"):
            return original_guided(*pos, **kw)
    pipe.cfg_guided_model_fn = guided
    original_decode = pipe.vae.decode
    def decode(*pos, **kw):
        with record.stage("vae_decode"):
            return original_decode(*pos, **kw)
    pipe.vae.decode = decode
    with record.stage("pipeline_total"):
        image = pipe(prompt=hp["prompt"], negative_prompt=hp["negative_prompt"], seed=hp["seed"],
                     height=hp["height"], width=hp["width"], cfg_scale=hp["cfg"],
                     num_inference_steps=hp["steps"], denoising_strength=hp["denoise"], sigma_shift=options["shift"])
    image.save(args.output / "image.png")

if __name__ == "__main__":
    args = parse_run("diffsynth")
    add_source("diffsynth")
    record = Recorder(args)
    try:
        run(args, record)
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish()
