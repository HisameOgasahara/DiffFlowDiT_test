import zipfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
destination = ROOT.parent / "sampling_synchro.zip"
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or relative.parts[0].startswith(".venv") or relative.parts[0] in {".git", ".tools", ".cache", "runs", "models"} or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == "runtime_run.json":
            continue
        archive.write(path, Path(ROOT.name) / relative)
with zipfile.ZipFile(destination) as archive:
    names = set(archive.namelist())
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for backend, record in manifest.items():
        for path in record["files_sha256"]:
            expected = f"{ROOT.name}/{backend}/upstream/{path}"
            if expected not in names:
                raise ValueError(f"압축에서 누락된 원본 파일: {expected}")
print(destination, f"{destination.stat().st_size / 2**20:.1f} MiB")
