import subprocess
import sys
from functools import wraps
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, add_source, parse_run, Recorder, make_noise, make_comfy_sigmas, write_json, read_json

def ensure_converted(args):
    manifest = read_json(args.models / "model_manifest.json")
    dtype = args.options["dtype"]
    commit = read_json(ROOT / "source_manifest.json")["diffusers"]["commit"]
    for component in ("transformer", "text_conditioner", "text_encoder", "vae"):
        marker = args.models / "diffusers_converted" / dtype / component / "conversion_source.json"
        expected = {"weights": manifest["weights"], "dtype": dtype, "source_commit": commit}
        if marker.exists() and read_json(marker) == expected:
            continue
        subprocess.run([sys.executable, str(Path(__file__).with_name("convert_weights.py")), "--models", str(args.models),
                        "--component", component, "--dtype", dtype], check=True)

def run(args, record):
    import torch
    from diffusers import AnimaAutoBlocks, AnimaTextConditioner, AutoencoderKLQwenImage, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler, ComponentsManager, ClassifierFreeGuidance
    from transformers import Qwen3Model, Qwen2Tokenizer, T5TokenizerFast
    hp, options = args.hp, args.options
    dtype = getattr(torch, options["dtype"])
    base = args.models / "diffusers_converted" / options["dtype"]
    with record.stage("load_models"):
        manager = ComponentsManager()
        if options["hf_offload"] == "auto":
            manager.enable_auto_cpu_offload(device="cuda:0")
        pipe = AnimaAutoBlocks().init_pipeline(components_manager=manager)
        components = {
            "transformer": CosmosTransformer3DModel.from_pretrained(base / "transformer", torch_dtype=dtype),
            "text_conditioner": AnimaTextConditioner.from_pretrained(base / "text_conditioner", torch_dtype=dtype),
            "text_encoder": Qwen3Model.from_pretrained(base / "text_encoder", torch_dtype=dtype),
            "vae": AutoencoderKLQwenImage.from_pretrained(base / "vae", torch_dtype=dtype),
            "tokenizer": Qwen2Tokenizer.from_pretrained(args.models / "tokenizers/qwen25_tokenizer"),
            "t5_tokenizer": T5TokenizerFast.from_pretrained(args.models / "tokenizers/t5_tokenizer"),
            "scheduler": FlowMatchEulerDiscreteScheduler(shift=options["shift"]),
            "guider": ClassifierFreeGuidance(guidance_scale=hp["cfg"]),
        }
        pipe.update_components(**components)
        if options["hf_offload"] == "group":
            from diffusers.hooks import apply_group_offloading
            for name in ("transformer", "text_conditioner", "text_encoder", "vae"):
                apply_group_offloading(getattr(pipe, name), onload_device=torch.device("cuda"),
                                      offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=False)
        elif options["hf_offload"] == "none":
            pipe.to("cuda")
        elif options["hf_offload"] != "auto":
            raise ValueError("hf_offload는 auto, group, none 중 하나여야 합니다.")
        if options["vae_tiling"]:
            pipe.vae.enable_tiling()
        for name in ("transformer", "text_encoder", "text_conditioner", "vae"):
            record.observe_model(name, getattr(pipe, name))
    record.data.update(effective_sampler="FlowMatchEulerDiscreteScheduler", effective_schedule="shifted FlowMatch grid",
                       requested_sampler=hp["sampler_name"], sampling_mismatch_with_png=True, offload=options["hf_offload"])
    scheduler = pipe.scheduler
    original_set = scheduler.set_timesteps
    @wraps(original_set)
    def set_timesteps(*pos, **kw):
        original_set(*pos, **kw)
        if args.mode == "matched_euler":
            grid = make_comfy_sigmas(hp["steps"], options["shift"], options["training_steps"])
            scheduler.sigmas = grid
            scheduler.timesteps = (grid[:-1] * scheduler.config.num_train_timesteps).to(kw.get("device") or "cpu")
            record.data["effective_schedule"] = "ComfyUI simple explicit grid"
        record.save("sigmas", scheduler.sigmas)
        write_json(args.output / "schedule.json", {"sigmas": scheduler.sigmas.tolist(), "model_timesteps": (scheduler.timesteps / scheduler.config.num_train_timesteps).tolist()})
    scheduler.set_timesteps = set_timesteps
    original_step = scheduler.step
    count = 0
    def step(model_output, timestep, sample, *pos, **kw):
        nonlocal count
        record.save("x_before", sample.squeeze(2), count)
        record.save("velocity", model_output.squeeze(2), count)
        sigma = scheduler.sigmas[count].to(sample.device)
        record.save("denoised", (sample.float() - sigma * model_output.float()).squeeze(2), count)
        if count == 0 and args.mode == "native":
            record.save("initial_noise", sample.squeeze(2))
        prediction = model_output.float() if args.mode == "matched_euler" else model_output
        result = original_step(prediction, timestep, sample, *pos, **kw)
        if count == hp["steps"] - 1:
            value = result[0] if isinstance(result, tuple) else result.prev_sample
            record.save("final_model_latent", value.squeeze(2))
        count += 1
        return result
    scheduler.step = step
    call = dict(prompt=hp["prompt"], negative_prompt=hp["negative_prompt"], height=hp["height"], width=hp["width"],
                num_inference_steps=hp["steps"], generator=torch.Generator("cpu").manual_seed(hp["seed"]),
                max_sequence_length=options["max_sequence_length"], output=["images", "qwen_prompt_embeds", "negative_qwen_prompt_embeds", "t5_input_ids", "negative_t5_input_ids"])
    if args.mode == "matched_euler":
        call["latents"] = make_noise(hp).unsqueeze(2)
        record.save("initial_noise", call["latents"].squeeze(2))
    with record.stage("pipeline_total"):
        result = pipe(**call)
    for label, key in (("prompt", "qwen_prompt_embeds"), ("negative_prompt", "negative_qwen_prompt_embeds")):
        record.save(f"condition_{label}", result[key])
    record.save("condition_prompt_t5xxl_ids", result["t5_input_ids"])
    record.save("condition_negative_prompt_t5xxl_ids", result["negative_t5_input_ids"])
    result["images"][0].save(args.output / "image.png")

if __name__ == "__main__":
    args = parse_run("diffusers")
    add_source("diffusers")
    # 모델 형식 변환은 추론 성능 측정에 포함하지 않습니다.
    ensure_converted(args)
    record = Recorder(args)
    try:
        run(args, record)
    except BaseException as error:
        record.finish(error)
        raise
    else:
        record.finish()
