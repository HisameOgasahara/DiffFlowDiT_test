"""Diffusers의 Anima 연산을 ComfyUI 기준으로 연결한다.

SPDX-License-Identifier: GPL-3.0-or-later
"""
from types import MethodType

import torch

from common.comfy_alignment import (
    bind_torch_rms_norm, denormalize_latent, install_qwen_numerics,
    normalize_pixels, pixels_to_image, tokenize_prompt, vae_attention_forward,
    vae_causal_conv_forward, vae_rms_norm, weight_and_pad_condition,
)


class CosmosAttention:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        from comfy.ldm.cosmos.predict2 import torch_attention_op
        from comfy.quant_ops import ck
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(context).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(context).unflatten(-1, (attn.heads, -1))
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            half = cos.shape[-1] // 2
            rope = torch.stack((cos[..., :half], -sin[..., :half],
                                sin[..., :half], cos[..., :half]), -1).unflatten(-1, (2, 2))
            rope = rope.unsqueeze(1).unsqueeze(0)
            q, k = ck.rms_rope_split_half(q, k, rope, attn.norm_q.weight.to(q),
                                         attn.norm_k.weight.to(k), attn.norm_q.eps)
        else:
            q, k = attn.norm_q(q), attn.norm_k(k)
        out = torch_attention_op(q, k, v)
        return attn.to_out[1](attn.to_out[0](out))


class TextAttention:
    def __call__(self, attn, hidden_states, attention_mask=None, encoder_hidden_states=None,
                 position_embeddings=None, encoder_position_embeddings=None):
        from comfy.ldm.anima.model import Attention
        attn.n_heads, attn.head_dim = attn.num_attention_heads, attn.attention_head_dim
        return Attention.forward(attn, hidden_states, mask=attention_mask, context=encoder_hidden_states,
                                 position_embeddings=position_embeddings,
                                 position_embeddings_context=encoder_position_embeddings)


def cosmos_block_forward(self, hidden_states, encoder_hidden_states, embedded_timestep, temb=None,
                         image_rotary_emb=None, extra_pos_emb=None, attention_mask=None,
                         controlnet_residual=None, latents=None, block_idx=None):
    compute_dtype = embedded_timestep.dtype
    if compute_dtype == torch.float16:
        hidden_states = hidden_states.float()
    if extra_pos_emb is not None:
        hidden_states = hidden_states + extra_pos_emb
    normalized, gate = self.norm1(hidden_states, embedded_timestep, temb)
    value = self.attn1(normalized.to(compute_dtype), image_rotary_emb=image_rotary_emb)
    hidden_states = torch.addcmul(hidden_states, gate.to(hidden_states), value.to(hidden_states))
    normalized, gate = self.norm2(hidden_states, embedded_timestep, temb)
    value = self.attn2(normalized.to(compute_dtype), encoder_hidden_states=encoder_hidden_states,
                       attention_mask=attention_mask)
    hidden_states = torch.addcmul(hidden_states, gate.to(hidden_states), value.to(hidden_states))
    normalized, gate = self.norm3(hidden_states, embedded_timestep, temb)
    value = self.ff(normalized.to(compute_dtype))
    return torch.addcmul(hidden_states, gate.to(hidden_states), value.to(hidden_states))


def install_components(components):
    from comfy.ldm.anima.model import RotaryEmbedding
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditionerAttention
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageRMS_norm, QwenImageAttentionBlock, QwenImageCausalConv3d,
    )
    transformer = components["transformer"]
    for module in transformer.modules():
        if isinstance(module, RMSNorm):
            bind_torch_rms_norm(module)
    for block in transformer.transformer_blocks:
        block.forward = MethodType(cosmos_block_forward, block)
        block.attn1.set_processor(CosmosAttention())
        block.attn2.set_processor(CosmosAttention())
    original_norm_out = transformer.norm_out.forward
    def norm_out(hidden_states, embedded_timestep, temb=None):
        return original_norm_out(hidden_states.to(embedded_timestep.dtype), embedded_timestep, temb)
    transformer.norm_out.forward = norm_out
    def return_float_prediction(module, args, output):
        if isinstance(output, tuple):
            return (output[0].float(), *output[1:])
        output.sample = output.sample.float()
        return output
    transformer.register_forward_hook(return_float_prediction)
    for module in components["text_conditioner"].modules():
        if isinstance(module, AnimaTextConditionerAttention):
            module.set_processor(TextAttention())
    rotary = components["text_conditioner"].rotary_emb
    rotary.inv_freq = RotaryEmbedding(rotary.inv_freq.numel() * 2).inv_freq.to(rotary.inv_freq.device)
    for module in components["vae"].modules():
        if isinstance(module, QwenImageRMS_norm):
            module.forward = MethodType(vae_rms_norm, module)
        elif isinstance(module, QwenImageAttentionBlock):
            module.forward = MethodType(vae_attention_forward, module)
        elif isinstance(module, QwenImageCausalConv3d):
            module.forward = MethodType(vae_causal_conv_forward, module)


def install_pipeline(record):
    from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep
    from diffusers.modular_pipelines.anima.before_denoise import AnimaTextConditioningStep
    from diffusers.modular_pipelines.anima.decoders import AnimaVaeDecoderStep, AnimaProcessImagesOutputStep
    from diffusers.modular_pipelines.anima.denoise import AnimaLoopBeforeDenoiser

    install_qwen_numerics()
    prepared = {}

    def encode_prompt(cls, components, prompt, negative_prompt=None, prepare_unconditional_embeds=True,
                      max_sequence_length=512, device=None, dtype=None):
        output = {}
        for prefix, text, label in (("", prompt, "prompt"),
                                    ("negative_", negative_prompt or "", "negative_prompt")):
            if prefix and not prepare_unconditional_embeds:
                for name in ("qwen_prompt_embeds", "qwen_attention_mask", "t5_input_ids", "t5_attention_mask"):
                    output[prefix + name] = None
                continue
            ids, mask, t5_ids, weights = tokenize_prompt(text, device, record, label)
            hidden = components.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            prepared[id(t5_ids)] = weights, label
            output.update({prefix + "qwen_prompt_embeds": hidden,
                           prefix + "qwen_attention_mask": mask,
                           prefix + "t5_input_ids": t5_ids,
                           prefix + "t5_attention_mask": torch.ones_like(t5_ids)})
            record.save(f"condition_{label}", hidden)
            record.save(f"condition_{label}_attention_mask", mask)
            record.save(f"condition_{label}_t5xxl_weights", weights.squeeze(0))
        return output

    def condition(components, qwen_prompt_embeds, qwen_attention_mask, t5_input_ids,
                  t5_attention_mask, device, conditioning_dtype, output_dtype):
        weights, label = prepared[id(t5_input_ids)]
        value = components.text_conditioner(
            source_hidden_states=qwen_prompt_embeds.to(device=device, dtype=conditioning_dtype),
            target_input_ids=t5_input_ids.to(device),
            target_attention_mask=None, source_attention_mask=None,
        )
        value = weight_and_pad_condition(value.to(output_dtype), weights)
        record.save(f"adapted_{label}", value)
        return value

    @torch.no_grad()
    def before_denoise(self, components, block_state, i, t):
        block_state.latent_model_input = block_state.latents.to(block_state.dtype)
        block_state.timestep = t.expand(block_state.latents.shape[0]).float() / components.scheduler.config.num_train_timesteps
        return components, block_state

    @torch.no_grad()
    def decode(self, components, state):
        block = self.get_block_state(state)
        latent = denormalize_latent(block.latents)
        record.save("final_vae_latent", latent.squeeze(2))
        block.images = components.vae.decode(latent.to(components.vae.dtype), return_dict=False)[0][:, :, 0]
        self.set_block_state(state, block)
        return components, state

    @torch.no_grad()
    def process_images(self, components, state):
        block = self.get_block_state(state)
        pixels = normalize_pixels(block.images)
        record.save("pixels", pixels)
        if block.output_type == "pil":
            block.images = [pixels_to_image(pixels)]
        elif block.output_type == "np":
            block.images = pixels.cpu().permute(0, 2, 3, 1).numpy()
        else:
            block.images = pixels
        self.set_block_state(state, block)
        return components, state

    AnimaTextEncoderStep.encode_prompt = classmethod(encode_prompt)
    AnimaTextConditioningStep._condition_prompt_embeds = staticmethod(condition)
    AnimaLoopBeforeDenoiser.__call__ = before_denoise
    AnimaVaeDecoderStep.__call__ = decode
    AnimaProcessImagesOutputStep.__call__ = process_images
