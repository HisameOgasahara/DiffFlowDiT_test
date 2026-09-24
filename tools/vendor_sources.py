"""Copy local core sources without changing their contents."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = {
    "comfyui": ("ComfyUI", ["comfy", "comfy_execution", "folder_paths.py", "node_helpers.py", "latent_preview.py", "cuda_malloc.py", "LICENSE", "requirements.txt"]),
    "diffusers": ("diffusers", ["src/diffusers", "scripts/convert_anima_to_diffusers.py", "scripts/convert_cosmos_to_diffusers.py", "LICENSE", "setup.py", "pyproject.toml"]),
    "diffsynth": ("diffsynth-studio", ["diffsynth", "LICENSE", "pyproject.toml"]),
}

def copy_sources(reference):
    manifest = {}
    for name, (repo, entries) in LAYOUT.items():
        source = reference / repo
        destination = ROOT / name / "upstream"
        records = {}
        for entry in entries:
            item = source / entry
            files = item.rglob("*") if item.is_dir() else [item]
            for path in files:
                if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                    continue
                relative = path.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                records[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        manifest[name] = {"repository": repo, "commit": commit, "files_sha256": records}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({name: len(record["files_sha256"]) for name, record in manifest.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    copy_sources(parser.parse_args().reference.resolve())
