"""Create one isolated Colab environment for each backend."""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def find_uv():
    import shutil
    found = shutil.which("uv")
    if found:
        return found
    # PyPI wheel includes the executable; no global installation is needed.
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    executable = tools / ("uv.exe" if sys.platform == "win32" else "uv")
    if executable.is_file():
        return str(executable)
    uv_version = json.loads((ROOT / "config/environment.json").read_text(encoding="utf-8"))["uv"]
    with urllib.request.urlopen(f"https://pypi.org/pypi/uv/{uv_version}/json", timeout=60) as response:
        files = json.load(response)["urls"]
    platform_tag = "win_amd64" if sys.platform == "win32" else "manylinux_2_17_x86_64"
    item = next(item for item in files if item["filename"].endswith(".whl") and platform_tag in item["filename"])
    wheel = tools / item["filename"]
    urllib.request.urlretrieve(item["url"], wheel)
    import hashlib
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != item["digests"]["sha256"]:
        raise ValueError("uv wheel SHA256 검증에 실패했습니다.")
    with zipfile.ZipFile(wheel) as archive:
        name = next(name for name in archive.namelist() if name.endswith("/uv.exe" if sys.platform == "win32" else "/uv"))
        executable = tools / ("uv.exe" if sys.platform == "win32" else "uv")
        executable.write_bytes(archive.read(name))
    executable.chmod(0o755)
    return str(executable)

def setup(backend, cpu=False):
    uv = find_uv()
    environment = ROOT / (f".venv-test-{backend}" if cpu else f".venv-{backend}")
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.11", str(environment)], check=True)
    profile = json.loads((ROOT / "config/environment.json").read_text(encoding="utf-8"))
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    f"torch=={profile['torch']}", f"torchvision=={profile['torchvision']}",
                    "--index-url", "https://download.pytorch.org/whl/" + ("cpu" if cpu else profile["torch_index"])], check=True)
    requirements = ROOT / backend / "requirements.txt"
    subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(requirements),
                    "-c", str(ROOT / "config/constraints.txt")], check=True)
    print("실행 Python:", python)
    return python

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["comfyui", "diffusers", "diffsynth"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    setup(args.backend, args.cpu)
