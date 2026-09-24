import argparse
import itertools
import json
from pathlib import Path

def compare_tensor(left, right):
    import torch
    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "left_shape": list(a.shape), "right_shape": list(b.shape)}
    a, b = a.double(), b.double()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"status": "nonfinite"}
    delta = a - b
    return {"status": "compared", "max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "relative_l2": (delta.norm() / a.norm().clamp_min(1e-12)).item(), "exact": torch.equal(a, b)}

def compare_runs(paths, destination):
    import numpy as np
    from PIL import Image, ImageDraw
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for left, right in itertools.combinations(paths, 2):
        hp_a = json.loads((left / "generation.json").read_text(encoding="utf-8"))
        hp_b = json.loads((right / "generation.json").read_text(encoding="utf-8"))
        model_a = json.loads((left / "models.json").read_text(encoding="utf-8"))
        model_b = json.loads((right / "models.json").read_text(encoding="utf-8"))
        ma = json.loads((left / "metrics.json").read_text(encoding="utf-8"))
        mb = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
        shared = sorted({p.name for p in left.glob("*.pt")} & {p.name for p in right.glob("*.pt")})
        tensors = {name: compare_tensor(left / name, right / name) for name in shared}
        row = {"left": left.name, "right": right.name, "same_generation": hp_a == hp_b,
               "same_weights": model_a["weights"] == model_b["weights"],
               "same_tokenizers": model_a.get("tokenizers") == model_b.get("tokenizers"),
               "trace_levels": [ma.get("trace"), mb.get("trace")],
               "source_commits": [ma.get("source_commit"), mb.get("source_commit")],
               "same_mode": ma["mode"] == mb["mode"], "samplers": [ma.get("effective_sampler"), mb.get("effective_sampler")],
               "statuses": [ma["status"], mb["status"]], "tensors": tensors}
        if (left / "image.png").exists() and (right / "image.png").exists():
            a, b = [np.asarray(Image.open(path / "image.png").convert("RGB"), dtype=np.float64) for path in (left, right)]
            if a.shape == b.shape:
                row["image_rmse_0_255"] = float(np.sqrt(np.mean((a - b) ** 2)))
                diff = np.clip(np.abs(a - b) * 4, 0, 255).astype(np.uint8)
                Image.fromarray(diff).save(destination / f"{left.name}--{right.name}-diff_x4.png")
        results.append(row)
    (destination / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    images = [(path.name, Image.open(path / "image.png").convert("RGB")) for path in paths if (path / "image.png").exists()]
    if images:
        width = min(600, max(image.width for _, image in images))
        previews = []
        for name, image in images:
            image.thumbnail((width, width))
            previews.append((name, image))
        height = max(image.height for _, image in previews) + 35
        canvas = Image.new("RGB", (width * len(previews), height), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, image) in enumerate(previews):
            draw.text((i * width + 5, 8), name, fill="black")
            canvas.paste(image, (i * width, 35))
        canvas.save(destination / "comparison.png")
    lines = ["# 실행 비교", "", "픽셀 오차는 이미지 품질 점수가 아닙니다. 동일 설정·동일 가중치 여부를 먼저 확인하세요.", "",
             "| 실행 | 모드 | 상태 | 총 시간(초) | GPU allocated 최대(GiB) | CPU RSS 최대(GiB) |", "|---|---|---|---:|---:|---:|"]
    for path in paths:
        m = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        peak = max((s["peak_allocated_bytes"] for s in m["stages"].values()), default=0)
        lines.append(f"| {path.name} | {m['mode']} | {m['status']} | {m.get('total_seconds', 0):.2f} | {peak / 2**30:.2f} | {m.get('peak_cpu_rss_bytes', 0) / 2**30:.2f} |")
    lines += ["", "텐서 추적을 켠 실행에는 CPU 복사·파일 저장 시간이 포함됩니다. 순수 성능은 trace=none 실행끼리 비교하세요.",
              "", "상세 오차는 comparison.json, 나란히 보기는 comparison.png, 차이 강조는 diff_x4.png를 확인하세요."]
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("비교 저장:", destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_runs(args.runs, args.output)
