import argparse
import shutil
from pathlib import Path
from common.runtime import ROOT, read_json, write_json, sha256

def prepare(destination, config):
    from huggingface_hub import HfApi, hf_hub_download
    hp = read_json(config)
    source = read_json(ROOT / "config/model_source.json")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "model_manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else None
    revision = old["revision"] if old else HfApi().model_info(source["repository"], revision=source["revision"]).sha
    manifest = {"repository": source["repository"], "revision": revision, "weights": {},
                "original_png_weight_hashes_known": False, "tokenizers": {}}
    for kind, name in hp["models"].items():
        target = destination / "weights" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        remote = source["folders"][kind] + "/" + name
        if not target.exists():
            downloaded = hf_hub_download(source["repository"], remote, revision=revision)
            shutil.copy2(downloaded, target)
        digest = sha256(target)
        if old and digest != old["weights"][kind]["sha256"]:
            raise ValueError(f"기존 모델 manifest와 해시가 다릅니다: {target}")
        manifest["weights"][kind] = {"filename": target.name, "sha256": digest, "bytes": target.stat().st_size}
    tokenizer_root = ROOT / "comfyui/upstream/comfy/text_encoders"
    for name in ("qwen25_tokenizer", "t5_tokenizer"):
        target = destination / "tokenizers" / name
        shutil.copytree(tokenizer_root / name, target, dirs_exist_ok=True)
        manifest["tokenizers"][name] = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob("*") if p.is_file()}
    write_json(manifest_path, manifest)
    print("공통 가중치와 토크나이저 준비 완료:", destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/generation.json")
    args = parser.parse_args()
    prepare(args.models, args.config)
