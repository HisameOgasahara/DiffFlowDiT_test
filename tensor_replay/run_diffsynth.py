"""ComfyUI 조건 이후의 원본 DiffSynth DiT/CFG/Euler/VAE 실행."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import parse_run, add_source, Recorder
from inputs import load_inputs, finish_image


def run(args, record):
    import torch
    from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig, model_fn_anima
    context, noise, sigmas = load_inputs(args, record)
    dtype = getattr(torch, args.options["dtype"])
    vram = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype, onload_device="cpu",
                preparing_dtype=dtype, preparing_device="cuda", computation_dtype=dtype, computation_device="cuda")
    with record.stage("load_models"):
        pipe = AnimaImagePipeline.from_pretrained(torch_dtype=dtype, device="cuda",
            model_configs=[ModelConfig(path=str(args.weights[k]), **vram) for k in ("dit", "vae")],
            tokenizer_config=None, tokenizer_t5xxl_config=None,
            vram_limit=torch.cuda.mem_get_info()[0]/2**30-args.options["vram_margin_gib"])
    pipe.scheduler.set_timesteps(args.hp["steps"])
    pipe.scheduler.sigmas = sigmas[:-1].clone()
    pipe.scheduler.timesteps = sigmas[:-1] * pipe.scheduler.num_train_timesteps
    record.data.update(effective_sampler="native DiffSynth FlowMatchScheduler Euler",
                       computation_replacements=False, initial_noise_cast=str(dtype))
    pipe.load_models_to_device(["dit"])
    x = noise.squeeze(2).to("cuda", dtype)
    positive = {"prompt_emb": context[1:2].to("cuda", dtype), "t5xxl_ids": None}
    negative = {"prompt_emb": context[0:1].to("cuda", dtype), "t5xxl_ids": None}
    record.observe_model("dit", pipe.dit)
    with torch.no_grad(), record.stage("pipeline_total"):
        for i, t in enumerate(pipe.scheduler.timesteps):
            record.save("x_before", x.unsqueeze(2), i)
            timestep = t.unsqueeze(0).to("cuda", dtype)
            pred = pipe.cfg_guided_model_fn(model_fn_anima, args.hp["cfg"], {"latents": x},
                                           positive, negative, dit=pipe.dit, timestep=timestep)
            record.save("velocity", pred.unsqueeze(2), i)
            record.save("denoised", (x.float()-sigmas[i].to(x.device)*pred.float()).unsqueeze(2), i)
            x = pipe.step(pipe.scheduler, latents=x, progress_id=i, noise_pred=pred)
            record.save("x_after", x.unsqueeze(2), i)
        record.save("final_model_latent", x.unsqueeze(2))
        pipe.load_models_to_device(["vae"])
        decoded = pipe.vae.decode(x.unsqueeze(2), device=pipe.device)
        record.save("decoded_pixels", decoded)
        image = pipe.vae_output_to_image(decoded.squeeze(2))
        pipe.load_models_to_device([])
    finish_image(image, x, args, record)


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
