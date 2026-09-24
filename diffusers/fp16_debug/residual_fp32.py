"""원본 블록의 잔차 경로만 FP32로 유지하는 선택적 dtype hook."""
import torch


def install_residual_fp32(transformer):
    handles = []
    if transformer.dtype != torch.float16:
        return handles

    def promote_residual(module, args, kwargs):
        if args:
            return (args[0].float(), *args[1:]), kwargs
        return args, dict(kwargs, hidden_states=kwargs["hidden_states"].float())

    def prepare_compute(module, args, output):
        normalized, gate = output
        # Attention/MLP 입력은 FP16, 곱셈과 잔차 합산은 FP32.
        return normalized.to(torch.float16), gate.float()

    def prepare_projection(module, args, output):
        return output.to(torch.float16)

    for block in transformer.transformer_blocks:
        handles.append(block.register_forward_pre_hook(promote_residual, with_kwargs=True))
        for norm in (block.norm1, block.norm2, block.norm3):
            handles.append(norm.register_forward_hook(prepare_compute))
    handles.append(transformer.norm_out.register_forward_hook(prepare_projection))
    return handles
