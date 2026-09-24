"""원본 호출의 입력·출력을 관찰하며 인자와 반환값은 그대로 전달한다."""
from contextlib import contextmanager
from functools import wraps

from common.runtime import write_json


@contextmanager
def trace_pipeline(pipe, record):
    import torch
    replacements, handles = [], []
    state = {"step": 0, "branch": "prompt", "branch_calls": 0, "text_calls": 0}

    def frame(value):
        # 저장 사본만 ComfyUI의 B,C,T,H,W 축 순서로 표현한다.
        return value.unsqueeze(2) if value.ndim == 4 else value

    def wrap(owner, name, observer):
        had_local = name in vars(owner)
        old_local = vars(owner).get(name)
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            return observer(original, *args, **kwargs)

        replacements.append((owner, name, had_local, old_local))
        setattr(owner, name, observed)

    def noise(original, *args, **kwargs):
        value = original(*args, **kwargs)
        record.save("initial_noise", frame(value))
        return value

    def schedule(original, *args, **kwargs):
        result = original(*args, **kwargs)
        sigmas = pipe.scheduler.sigmas
        record.save("sigmas", torch.cat([sigmas, sigmas.new_zeros(1)]))
        write_json(record.args.output / "schedule.json", {
            "sigmas": sigmas.tolist() + [0.0],
            "scheduler_timesteps": pipe.scheduler.timesteps.tolist(),
            "model_timesteps_source": "dit_*_input_timesteps_*.pt (actual model inputs)",
            "terminal_zero_appended_for_comparison": True,
        })
        return result

    def unit(original, unit, pipeline, shared, positive, negative):
        with record.stage(type(unit).__name__):
            result = original(unit, pipeline, shared, positive, negative)
        if type(unit).__name__ == "AnimaUnit_PromptEmbedder":
            for label, values in zip(("prompt", "negative_prompt"), result[1:]):
                record.save(f"condition_{label}", values.get("prompt_emb"))
                ids = values.get("t5xxl_ids")
                record.save(f"condition_{label}_t5xxl_ids", ids.squeeze(0) if ids is not None else None)
        return result

    def guided(original, *args, **kwargs):
        state.update(step=kwargs["progress_id"], branch_calls=0)
        with record.stage(f"denoise_step_{state['step']:03d}"):
            return original(*args, **kwargs)

    def model(original, *args, **kwargs):
        state["branch"] = "prompt" if state["branch_calls"] == 0 else "negative_prompt"
        state["branch_calls"] += 1
        value = original(*args, **kwargs)
        record.save(f"velocity_{state['branch']}", frame(value), state["step"])
        return value

    def step(original, scheduler, latents, progress_id, noise_pred, **kwargs):
        record.save("x_before", frame(latents), progress_id)
        record.save("velocity", frame(noise_pred), progress_id)
        if record.args.trace != "none" and progress_id in record.selected:
            # 비교용 진단값이며 원본 solver에는 전달하지 않는다.
            sigma = scheduler.sigmas[progress_id].to(latents.device)
            diagnostic = latents.float() - sigma.float() * noise_pred.float()
            record.save("denoised", frame(diagnostic), progress_id)
        result = original(scheduler, latents, progress_id, noise_pred, **kwargs)
        record.save("x_after", frame(result), progress_id)
        if progress_id == record.args.hp["steps"] - 1:
            record.save("final_model_latent", frame(result))
        return result

    def decode(original, *args, **kwargs):
        with record.stage("vae_decode"):
            result = original(*args, **kwargs)
        record.save("decoded_pixels", result)
        return result

    def image(original, value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if record.args.trace != "none":
            import numpy as np
            # 원본 픽셀 변환이 끝난 실제 PNG 값. 양자화 전 출력은 decoded_pixels에 저장한다.
            pixels = torch.from_numpy(np.array(result, copy=True)).permute(2, 0, 1).unsqueeze(0)
            record.save("pixels", pixels.float() / 255)
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
        record.save(f"condition_{label}_attention_mask", kwargs.get("attention_mask"))

    def dit_input(module, args, kwargs):
        for key, value in kwargs.items():
            record.save(f"dit_{state['branch']}_input_{key}", value, state["step"])

    def adapter_output(module, args, output):
        record.save(f"text_adapter_{state['branch']}_output", output, state["step"])

    def block_input(module, args, kwargs):
        value = kwargs.get("crossattn_emb")
        record.save(f"adapted_{state['branch']}", value, state["step"])

    def vae_input(module, args):
        record.save("final_vae_latent", args[0])

    try:
        wrap(pipe, "generate_noise", noise)
        wrap(pipe.scheduler, "set_timesteps", schedule)
        wrap(pipe, "unit_runner", unit)
        wrap(pipe, "cfg_guided_model_fn", guided)
        wrap(pipe, "model_fn", model)
        wrap(pipe, "step", step)
        wrap(pipe.vae, "decode", decode)
        wrap(pipe, "vae_output_to_image", image)
        if record.args.trace != "none":
            handles.append(pipe.text_encoder.register_forward_pre_hook(text_input, with_kwargs=True))
            handles.append(pipe.dit.register_forward_pre_hook(dit_input, with_kwargs=True))
            handles.append(pipe.dit.llm_adapter.register_forward_hook(adapter_output))
            handles.append(pipe.dit.blocks[0].register_forward_pre_hook(block_input, with_kwargs=True))
            handles.append(pipe.vae.model.conv2.register_forward_pre_hook(vae_input))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, had_local, value in reversed(replacements):
            if had_local:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
