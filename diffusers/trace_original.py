"""Diffusers 원본 호출의 관찰용 래퍼와 반환값을 바꾸지 않는 hook."""
from contextlib import contextmanager
from functools import wraps

from common.runtime import write_json


@contextmanager
def trace_pipeline(pipe, record):
    replacements, handles = [], []
    state = {"step": 0, "text_calls": 0, "adapter_calls": 0}

    def wrap(owner, name, observer):
        had_local, old_local = name in vars(owner), vars(owner).get(name)
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            return observer(original, *args, **kwargs)

        replacements.append((owner, name, had_local, old_local))
        setattr(owner, name, observed)

    def schedule(original, *args, **kwargs):
        result = original(*args, **kwargs)
        record.save("sigmas", pipe.scheduler.sigmas)
        write_json(record.args.output / "schedule.json", {
            "sigmas": pipe.scheduler.sigmas.tolist(),
            "scheduler_timesteps": pipe.scheduler.timesteps.tolist(),
            "model_timesteps_source": "dit_*_input_timestep_*.pt (actual model inputs)",
        })
        return result

    def step(original, model_output, timestep, sample, *args, **kwargs):
        i = state["step"]
        if i == 0:
            record.save("initial_noise", sample)
        record.save("x_before", sample, i)
        record.save("velocity", model_output, i)
        if record.args.trace != "none" and i in record.selected:
            sigma = pipe.scheduler.sigmas[i].to(sample.device)
            record.save("denoised", sample.float() - sigma.float() * model_output.float(), i)
        with record.stage(f"scheduler_step_{i:03d}"):
            result = original(model_output, timestep, sample, *args, **kwargs)
        value = result[0] if isinstance(result, tuple) else result.prev_sample
        record.save("x_after", value, i)
        if i == record.args.hp["steps"] - 1:
            record.save("final_model_latent", value)
        state["step"] += 1
        return result

    def decode(original, z, *args, **kwargs):
        record.save("final_vae_latent", z)
        with record.stage("vae_decode"):
            result = original(z, *args, **kwargs)
        record.save("decoded_pixels", result[0] if isinstance(result, tuple) else result.sample)
        return result

    def text_input(module, args, kwargs):
        label = "prompt" if state["text_calls"] == 0 else "negative_prompt"
        state["text_calls"] += 1
        tokens = {}
        for key in ("input_ids", "attention_mask"):
            value = kwargs.get(key)
            record.save(f"text_encoder_{label}_{key}", value)
            if value is not None:
                tokens[key] = value.detach().cpu().tolist()
        write_json(record.args.output / f"tokens_{label}.json", tokens)

    def adapter_output(module, args, kwargs, output):
        label = "prompt" if state["adapter_calls"] == 0 else "negative_prompt"
        state["adapter_calls"] += 1
        for key, value in kwargs.items():
            record.save(f"text_adapter_{label}_input_{key}", value)
        record.save(f"text_adapter_{label}_output", output)

    def dit_input(module, args, kwargs):
        label = "prompt" if pipe.guider.is_conditional else "negative_prompt"
        for key, value in kwargs.items():
            record.save(f"dit_{label}_input_{key}", value, state["step"])

    def dit_output(module, args, kwargs, output):
        label = "prompt" if pipe.guider.is_conditional else "negative_prompt"
        record.save(f"velocity_{label}", output[0] if isinstance(output, tuple) else output.sample, state["step"])

    try:
        wrap(pipe.scheduler, "set_timesteps", schedule)
        wrap(pipe.scheduler, "step", step)
        wrap(pipe.vae, "decode", decode)
        if record.args.trace != "none":
            handles.append(pipe.text_encoder.register_forward_pre_hook(text_input, with_kwargs=True))
            handles.append(pipe.text_conditioner.register_forward_hook(adapter_output, with_kwargs=True))
            handles.append(pipe.transformer.register_forward_pre_hook(dit_input, with_kwargs=True))
            handles.append(pipe.transformer.register_forward_hook(dit_output, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, had_local, value in reversed(replacements):
            if had_local:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
