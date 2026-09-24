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
    profile = json.loads((ROOT / "config/environment.json").read_text(encoding="utf-8"))
    baseline = profile["baseline"]
    if not cpu and python.exists():
        actual = subprocess.check_output([str(python), "-c", "import sys; print(sys.version.split()[0])"], text=True).strip()
        if actual != baseline["python"]:
            raise RuntimeError("기존 가상환경의 Python이 기준과 다릅니다. 새 Colab 런타임에서 실행하세요.")
    if not python.exists():
        command = [uv, "venv", "--python", "3.11" if cpu else baseline["python"]]
        if not cpu:
            command += ["--seed", "--system-site-packages"]
        subprocess.run(command + [str(environment)], check=True)
    if cpu:
        subprocess.run([uv, "pip", "install", "--python", str(python),
                    f"torch=={profile['torch']}", f"torchvision=={profile['torchvision']}",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    else:
        subprocess.run([uv, "pip", "install", "--python", str(python),
                        f"torch=={baseline['torch']}", f"torchvision=={baseline['torchvision']}",
                        "--torch-backend", baseline["torch_backend"]], check=True)
    constraints = ["-c", str(ROOT / "config/constraints.txt")]
    if not cpu:
        baseline_constraints = environment / "baseline_constraints.txt"
        baseline_constraints.write_text("".join(f"{name}=={baseline[name]}\n" for name in ("torch", "torchvision")), encoding="utf-8")
        constraints += ["-c", str(baseline_constraints)]
    requirements = ROOT / backend / "requirements.txt"
    subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(requirements),
                    *constraints], check=True)
    if not cpu:
        actual = subprocess.check_output([str(python), "-c", "import sys,torch; print(sys.version.split()[0]); print(torch.__version__)"], text=True).splitlines()
        if actual != [baseline["python"], baseline["torch"]]:
            raise RuntimeError(f"설치된 환경이 baseline과 다릅니다: {actual}")
    print("실행 Python:", python)
    return python

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["comfyui", "diffusers", "diffsynth"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    setup(args.backend, args.cpu)
