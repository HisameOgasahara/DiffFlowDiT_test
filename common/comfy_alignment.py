"""ComfyUI Anima의 토큰·수치 처리 규칙을 각 구현의 모델에 적용한다.

ComfyUI 코어의 함수를 재사용한다. 모델과 가중치 로더는 각 구현에 남긴다.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from types import MethodType

import torch
import torch.nn.functional as F

from common.runtime import add_source, write_json


def load_comfy_policy(weights):
    add_source("comfyui")
    from safetensors import safe_open
    from comfy import model_management as mm
    from comfy.utils import weight_dtype, calculate_parameters

    # sd.load_diffusion_model과 같은 가중치 dtype·파라미터 수로 자동 선택한다.
    names = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    with safe_open(str(weights["dit"]), framework="pt", device="cpu") as checkpoint:
        metadata = {
            key: torch.empty(checkpoint.get_slice(key).get_shape(),
                             dtype=names[checkpoint.get_slice(key).get_dtype()], device="meta")
            for key in checkpoint.keys()
        }
    allowed = [torch.bfloat16, torch.float16, torch.float32]
    stored = mm.unet_dtype(model_params=calculate_parameters(metadata),
                          supported_dtypes=allowed, weight_dtype=weight_dtype(metadata))
    compute = mm.unet_manual_cast(stored, mm.get_torch_device(), allowed) or stored
    # sd1_clip.SDClipModel.forward는 Qwen의 embedding과 연산을 FP32로 전달한다.
    # sd.VAE의 Wan 2.1 경로에 지정된 working_dtypes와 같은 순서다.
    return {"transformer": compute, "text_conditioner": compute,
            "text_encoder": torch.float32,
            "vae": mm.vae_dtype(mm.vae_device(), allowed)}


def tokenize_prompt(prompt, device, record=None, label=None):
    from comfy.text_encoders.anima import AnimaTokenizer
    tokenizer = AnimaTokenizer()
    tokens = tokenizer.tokenize_with_weights(prompt)
    if record is not None:
        write_json(record.args.output / f"tokens_{label}.json", tokens)
    qwen = tokens["qwen3_06b"][0]
    t5 = tokens["t5xxl"][0]
    ids = torch.tensor([[item[0] for item in qwen]], device=device, dtype=torch.long)
    # SDClipModel.process_tokens의 leading pad와 end/pad 처리 순서.
    pad = tokenizer.qwen3_06b.pad_token
    values = []
    eos = False
    left_pad = bool(qwen and qwen[0][0] == pad)
    for token, _ in qwen:
        if eos or (left_pad and token == pad):
            values.append(0)
        else:
            values.append(1)
            left_pad = False
        if not eos and token == pad and not left_pad:
            values[-1] = 0
            eos = True
    mask = torch.tensor([values], device=device, dtype=torch.long)
    t5_ids = torch.tensor([[item[0] for item in t5]], device=device, dtype=torch.long)
    weights = torch.tensor([[item[1] for item in t5]], device=device, dtype=torch.float32)
    return ids, mask, t5_ids, weights


def _qwen_forward(self, input_ids=None, attention_mask=None, **kwargs):
    from comfy.text_encoders.llama import precompute_freqs_cis
    from comfy.ldm.modules.attention import optimized_attention_for_device
    from transformers.modeling_outputs import BaseModelOutputWithPast

    x = self.embed_tokens(input_ids).float()
    length = x.shape[1]
    positions = torch.arange(length, device=x.device).unsqueeze(0)
    freqs = precompute_freqs_cis(self.config.head_dim, positions,
                                self.config.rope_theta, device=x.device)
    mask = 1.0 - attention_mask.to(x.dtype).reshape(x.shape[0], 1, 1, length).expand(-1, 1, length, -1)
    mask = mask.masked_fill(mask.bool(), torch.finfo(x.dtype).min / 4)
    if length > 1:
        mask = mask + torch.full((length, length), torch.finfo(x.dtype).min / 4,
                                dtype=x.dtype, device=x.device).triu_(1)
    attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
    for layer in self.layers:
        x, _ = layer(x=x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention)
    x = self.norm(x)
    return BaseModelOutputWithPast(last_hidden_state=x, hidden_states=(x,))


def _qwen_attention(self, **kwargs):
    from comfy.text_encoders.llama import Attention
    self.merged_qkv = False
    self.num_heads = self.config.num_attention_heads
    self.num_kv_heads = self.config.num_key_value_heads
    return Attention.forward(self, **kwargs)


def _qwen_norm(self, x):
    from comfy.rmsnorm import rms_norm
    return rms_norm(x, self.weight, self.variance_epsilon)


def install_qwen_numerics():
    # 모델 생성 전에 적용하므로 각 프레임워크의 offload hook을 보존한다.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer, Qwen3Attention, Qwen3RMSNorm
    from comfy.text_encoders.llama import TransformerBlock
    Qwen3Model.forward = _qwen_forward
    Qwen3DecoderLayer.forward = TransformerBlock.forward
    Qwen3Attention.forward = _qwen_attention
    Qwen3RMSNorm.forward = _qwen_norm


def weight_and_pad_condition(condition, weights):
    # Anima.preprocess_text_embeds와 같은 순서: 가중치 곱 → 최소 512 길이 zero pad.
    count = weights.shape[-1]
    condition = condition[:, :count] * weights.to(condition).unsqueeze(-1)
    return F.pad(condition, (0, 0, 0, max(0, 512 - count)))


def denormalize_latent(latent):
    from comfy.latent_formats import Wan21
    return Wan21().process_out(latent.float())


def normalize_pixels(decoded):
    # comfy.sd.VAE.decode: 디코더 출력은 FP32로 변환한 다음 process_output.
    return ((decoded.float() + 1.0) / 2.0).clamp(0, 1)


def vae_rms_norm(self, x):
    bias = self.bias.to(x) if torch.is_tensor(self.bias) else (self.bias or 0)
    return F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma.to(x) + bias


def bind_torch_rms_norm(module):
    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight.to(x), self.eps)
    module.forward = MethodType(forward, module)


def vae_attention_forward(self, x, *args, **kwargs):
    from comfy.ldm.modules.diffusionmodules.model import vae_attention
    if not hasattr(self, "optimized_attention"):
        self.optimized_attention = vae_attention()
    batch, channels, time, height, width = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    q, k, v = self.to_qkv(self.norm(flat)).chunk(3, dim=1)
    out = self.proj(self.optimized_attention(q, k, v))
    return out.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4) + x


def vae_causal_conv_forward(self, x, cache_x=None):
    from comfy.ops import NVIDIA_MEMORY_CONV_BUG_WORKAROUND
    weight = self.weight.to(x)
    bias = self.bias.to(x) if self.bias is not None else None
    spatial_padding = (0, self._padding[2], self._padding[0])
    if cache_x is None and x.shape[2] == 1:
        weight = weight[:, :, -x.shape[2]:]
    else:
        remaining = self._padding[4]
        if cache_x is not None:
            cache_x = cache_x.to(x)
            remaining = max(0, remaining - cache_x.shape[2])
            x = torch.cat((cache_x, x), dim=2)
        x = F.pad(x, (0, 0, 0, 0, remaining, 0))
    if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
        out = torch.cudnn_convolution(x, weight, spatial_padding, self.stride, self.dilation,
                                     self.groups, benchmark=False, deterministic=False, allow_tf32=True)
        if bias is not None:
            out += bias.reshape(1, -1, 1, 1, 1)
        return out
    return F.conv3d(x, weight, bias, self.stride, spatial_padding, self.dilation, self.groups)


def pixels_to_image(pixels):
    import numpy as np
    from PIL import Image
    array = pixels[0].detach().float().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))
