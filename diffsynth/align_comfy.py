"""DiffSynth Anima의 조건·정밀도·VAE 경로를 ComfyUI에 맞춘다.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from types import MethodType

import torch

from common.comfy_alignment import (
    denormalize_latent, install_qwen_numerics, normalize_pixels, pixels_to_image,
    tokenize_prompt, vae_attention_forward, vae_causal_conv_forward, vae_rms_norm,
    weight_and_pad_condition,
)


def compute_qkv(self, x, context=None, rope_emb=None):
    from comfy.quant_ops import ck
    context = x if context is None else context
    q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
    k = self.k_proj(context).unflatten(-1, (self.n_heads, self.head_dim))
    v = self.v_proj(context).unflatten(-1, (self.n_heads, self.head_dim))
    if self.is_selfattn and rope_emb is not None:
        qnorm = getattr(self.q_norm, "module", self.q_norm)
        knorm = getattr(self.k_norm, "module", self.k_norm)
        q, k = ck.rms_rope_split_half(q, k, rope_emb, qnorm.weight.to(q), knorm.weight.to(k), qnorm.eps)
    else:
        q, k = self.q_norm(q), self.k_norm(k)
    return q, k, v


def install_models():
    from comfy.ldm.cosmos.predict2 import Block, torch_attention_op
    from diffsynth.models import anima_dit, wan_video_vae
    install_qwen_numerics()
    anima_dit.Block.forward = Block.forward
    anima_dit.Attention.compute_qkv = compute_qkv
    anima_dit.torch_attention_op = torch_attention_op
    wan_video_vae.RMS_norm.forward = vae_rms_norm
    wan_video_vae.AttentionBlock.forward = vae_attention_forward
    wan_video_vae.CausalConv3d.forward = vae_causal_conv_forward


def install_pipeline(pipe, record, policy):
    from comfy.ldm.anima.model import RotaryEmbedding
    from diffsynth.pipelines.anima_image import AnimaUnit_PromptEmbedder
    from diffsynth.diffusion.base_pipeline import PipelineUnit

    rotary = pipe.dit.llm_adapter.rotary_emb
    rotary.inv_freq = RotaryEmbedding(rotary.inv_freq.numel() * 2).inv_freq.to(rotary.inv_freq.device)
    from diffsynth.models.anima_dit import VideoRopePosition3DEmb
    for module in pipe.dit.modules():
        if isinstance(module, VideoRopePosition3DEmb):
            for name in ("dim_spatial_range", "dim_temporal_range"):
                old = getattr(module, name)
                setattr(module, name, torch.arange(0, old.numel() * 2, 2, device=old.device).float() / (old.numel() * 2))

    prompt_count = 0
    def encode_prompt(unit, pipeline, prompt):
        nonlocal prompt_count
        pipeline.load_models_to_device(["text_encoder"])
        label = "prompt" if prompt_count == 0 else "negative_prompt"
        prompt_count += 1
        ids, mask, t5_ids, weights = tokenize_prompt(prompt, pipeline.device, record, label)
        hidden = pipeline.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        record.save(f"condition_{label}_attention_mask", mask)
        record.save(f"condition_{label}_t5xxl_weights", weights.squeeze(0))
        return {"prompt_emb": hidden, "t5xxl_ids": t5_ids, "t5xxl_weights": weights}

    for unit in pipe.units:
        if isinstance(unit, AnimaUnit_PromptEmbedder):
            unit.process = MethodType(encode_prompt, unit)

    class PrepareCondition(PipelineUnit):
        def __init__(self):
            super().__init__(seperate_cfg=True,
                             input_params_posi={"prompt_emb": "prompt_emb", "t5xxl_ids": "t5xxl_ids", "t5xxl_weights": "t5xxl_weights"},
                             input_params_nega={"prompt_emb": "prompt_emb", "t5xxl_ids": "t5xxl_ids", "t5xxl_weights": "t5xxl_weights"},
                             output_params=("prompt_emb", "t5xxl_ids"), onload_model_names=("dit",))
            self.count = 0

        def process(self, pipeline, prompt_emb, t5xxl_ids, t5xxl_weights):
            pipeline.load_models_to_device(["dit"])
            value = pipeline.dit.llm_adapter(prompt_emb.to(policy["transformer"]), t5xxl_ids)
            value = weight_and_pad_condition(value, t5xxl_weights)
            label = "prompt" if self.count == 0 else "negative_prompt"
            self.count += 1
            record.save(f"adapted_{label}", value)
            return {"prompt_emb": value, "t5xxl_ids": None}

    pipe.units.append(PrepareCondition())

    def model_fn(dit, latents, timestep, prompt_emb, progress_id, **kwargs):
        # 원본 pipeline의 FP16 timestep 변환 전에 보관된 FP32 시간표를 사용한다.
        t = pipe.scheduler.timesteps[progress_id].reshape(1).to(pipe.device).float() / record.args.options["training_steps"]
        return dit(x=latents.to(policy["transformer"]).unsqueeze(2), timesteps=t,
                   context=prompt_emb.to(policy["transformer"]), t5xxl_ids=None).squeeze(2).float()
    pipe.model_fn = model_fn

    def decode(latents, device, **kwargs):
        latent = denormalize_latent(latents)
        record.save("final_vae_latent", latent.squeeze(2))
        # 정규화는 위에서 FP32로 한 번만 되돌리고 VAE의 기존 decoder를 호출한다.
        identity = [torch.zeros_like(pipe.vae.mean), torch.ones_like(pipe.vae.std)]
        return pipe.vae.model.decode(latent.to(device=device, dtype=policy["vae"]), identity)
    pipe.vae.decode = decode

    def output_image(decoded, **kwargs):
        pixels = normalize_pixels(decoded)
        record.save("pixels", pixels)
        return pixels_to_image(pixels)
    pipe.vae_output_to_image = output_image
