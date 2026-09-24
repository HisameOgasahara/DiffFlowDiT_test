"""출처가 고정된 ComfyUI 입력 묶음을 검증하고 읽는다."""
from common.runtime import ROOT, read_json, sha256, write_json


def load_inputs(args, record):
    import torch
    folder = ROOT / "tensor_replay/comfy_inputs"
    manifest = read_json(folder / "manifest.json")
    for name, info in manifest["files"].items():
        if sha256(folder / name) != info["sha256"]:
            raise ValueError(f"ComfyUI 입력 해시 불일치: {name}")
    if args.hp != read_json(folder / "generation.json"):
        raise ValueError("이 실험은 입력 묶음의 generation.json을 사용해야 합니다.")
    if read_json(args.models / "model_manifest.json")["weights"] != read_json(folder / "models.json")["weights"]:
        raise ValueError("ComfyUI와 모델 가중치가 다릅니다.")
    load = lambda name: torch.load(folder / name, map_location="cpu", weights_only=True)
    context, noise, sigmas = load("context.pt"), load("initial_noise.pt"), load("sigmas.pt")
    assert context.shape == (2, 512, 1024) and len(sigmas) == args.hp["steps"] + 1
    assert sigmas[-1] == 0 and torch.all(sigmas[:-1] > sigmas[1:])
    for value in (context, noise, sigmas):
        assert torch.isfinite(value).all()
    write_json(args.output / "input_manifest.json", manifest)
    record.data.update(mode="comfy_tensor_replay", text_encoding_skipped=True,
                       text_adapter_skipped=True, effective_schedule="saved ComfyUI Euler sigmas")
    record.save("initial_noise", noise)
    record.save("sigmas", sigmas)
    record.save("adapted_prompt", context[1:2])
    record.save("adapted_negative_prompt", context[0:1])
    return context, noise, sigmas


def finish_image(image, latent, args, record):
    import torch
    import numpy as np
    from PIL import Image
    image.save(args.output / "image.png")
    pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255
    record.save("pixels", pixels)
    reference = torch.load(ROOT / "tensor_replay/comfy_inputs/reference_latent.pt", weights_only=True)
    error = (latent.detach().cpu().float().reshape_as(reference) - reference).norm() / reference.norm()
    ref_image = np.asarray(Image.open(ROOT / "tensor_replay/comfy_inputs/reference.png")).astype(float)
    write_json(args.output / "comparison.json", {"latent_relative_l2": error.item(),
        "image_rmse_255": float(np.mean((np.asarray(image).astype(float)-ref_image)**2)**0.5)})
