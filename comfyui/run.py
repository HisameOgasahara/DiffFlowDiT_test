import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import add_source, parse_run, Recorder, make_comfy_sigmas, write_json, save_image

def run(args, record):
    import torch
    import comfy.sd
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import comfy.model_management as mm
    from comfy.ldm.modules import attention
    from headless_init import initialize_devices

    hp, options = args.hp, args.options
    record.data["dynamic_vram_enabled"] = initialize_devices()
    record.data["attention_dispatch"] = attention.optimized_attention.__name__
    record.data["memory_policy"] = str(mm.vram_state)
    record.data["effective_sampler"] = "euler" if args.mode == "matched_euler" else hp["sampler_name"]
    record.data["effective_schedule"] = hp["scheduler"]
    record.data["precision_policy"] = options["comfy_dtype"]
    with torch.inference_mode():
        with record.stage("text_encode"):
            clip = comfy.sd.load_clip([str(args.weights["text_encoder"])], clip_type=comfy.sd.CLIPType[hp["clip_type"].upper()])
            encoder = getattr(clip.cond_stage_model, clip.cond_stage_model.clip).transformer
            record.observe_model("text_encoder", encoder)
            conditions = []
            for name in ("prompt", "negative_prompt"):
                tokens = clip.tokenize(hp[name])
                write_json(args.output / f"tokens_{name}.json", tokens)
                cond = clip.encode_from_tokens_scheduled(tokens)
                record.save(f"condition_{name}", cond[0][0])
                for key, value in cond[0][1].items():
                    record.save(f"condition_{name}_{key}", value)
                conditions.append(cond)
            mm.unload_all_models()
            del clip, encoder

        with record.stage("load_dit"):
            model = comfy.sd.load_diffusion_model(str(args.weights["dit"]))
            record.observe_model("dit", model.model.diffusion_model)
            record.observe_model("text_adapter", model.model.diffusion_model.llm_adapter)
            sampling = model.get_model_object("model_sampling")
            record.data["model_sampling"] = {"class": type(sampling).__name__, "shift": sampling.shift, "multiplier": sampling.multiplier}
        with record.stage("sampling"):
            latent = torch.zeros((1, 4, hp["height"] // 8, hp["width"] // 8), device=mm.intermediate_device())
            latent = comfy.sample.fix_empty_latent_channels(model, latent)
            noise = comfy.sample.prepare_noise(latent, hp["seed"])
            record.save("initial_noise", noise)
            sigmas = comfy.samplers.calculate_sigmas(sampling, hp["scheduler"], hp["steps"])
            if args.mode == "matched_euler":
                expected = make_comfy_sigmas(hp["steps"], options["shift"], options["training_steps"])
                if not torch.allclose(sigmas.cpu(), expected, atol=1e-6, rtol=1e-6):
                    raise ValueError("원본 ComfyUI 시간표와 공통 시간표가 다릅니다. shift 설정을 확인하세요.")
            record.save("sigmas", sigmas)
            write_json(args.output / "schedule.json", {"sigmas": sigmas.tolist(), "model_timesteps": sampling.timestep(sigmas).tolist()})
            def callback(step, denoised, x, total_steps):
                record.save("x_before", x, step)
                record.save("denoised", denoised, step)
            from comfy.k_diffusion import sampling as kd
            sampler_function = "sample_" + record.data["effective_sampler"]
            original_sampler = getattr(kd, sampler_function)
            def observed_sampler(model_fn, x, sigmas, **kwargs):
                previous_callback = kwargs.get("callback")
                def observe(info):
                    sigma = info.get("sigma_hat", info["sigma"])
                    record.save("evaluated_sigma", sigma, info["i"])
                    record.save("velocity", (info["x"].float() - info["denoised"].float()) / sigma, info["i"])
                    if previous_callback:
                        previous_callback(info)
                kwargs["callback"] = observe
                return original_sampler(model_fn, x, sigmas, **kwargs)
            setattr(kd, sampler_function, observed_sampler)
            if args.mode == "native" and hp["sampler_name"] == "er_sde":
                original_noise_sampler = kd.default_noise_sampler
                def traced_noise_sampler(x, seed=None):
                    sampler = original_noise_sampler(x, seed=seed)
                    count = 0
                    def draw(sigma, sigma_next):
                        nonlocal count
                        value = sampler(sigma, sigma_next)
                        record.save("sde_noise", value, count)
                        count += 1
                        return value
                    return draw
                kd.default_noise_sampler = traced_noise_sampler
            samples = comfy.sample.sample(model, noise, hp["steps"], hp["cfg"], record.data["effective_sampler"],
                                          hp["scheduler"], conditions[0], conditions[1], latent,
                                          denoise=hp["denoise"], sigmas=sigmas, callback=callback, seed=hp["seed"])
            record.save("final_model_latent", model.model.process_latent_in(samples))
            record.save("final_vae_latent", samples)
            mm.unload_all_models()
            del model, conditions
        with record.stage("vae_decode"):
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.weights["vae"])))
            record.observe_model("vae", vae.first_stage_model)
            if options["vae_tiling"]:
                pixels = vae.decode_tiled(samples)
            else:
                pixels = vae.decode(samples)
            # ComfyUI nodes.VAEDecode: combine batch and frame dimensions.
            if pixels.ndim == 5:
                pixels = pixels.reshape(-1, *pixels.shape[-3:])
            record.save("pixels", pixels.movedim(-1, 1))
            save_image(pixels, args.output / "image.png", channel_last=True)
        record.data["ui_started"] = False

if __name__ == "__main__":
    args = parse_run("comfyui")
    add_source("comfyui")
    import comfy.options
    comfy.options.enable_args_parsing()
    flags = [sys.argv[0], "--disable-all-custom-nodes", "--disable-api-nodes"]
    if args.options["comfy_dtype"] == "float16":
        flags += ["--fp16-unet", "--fp16-text-enc", "--fp16-vae"]
    if args.options["comfy_memory"] == "low":
        flags += ["--lowvram"]
    sys.argv = flags
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from headless_init import initialize_before_torch
    initialize_before_torch()
    record = Recorder(args)
    try:
        run(args, record)
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish()
