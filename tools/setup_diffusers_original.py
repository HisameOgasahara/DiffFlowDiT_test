"""원본 Diffusers용 독립 환경을 기존 환경 JSON으로 설치한다."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, read_json
from tools.setup_environment import find_uv


def setup():
    uv = find_uv()
    baseline = read_json(ROOT / "config/environment.json")["baseline"]
    environment = ROOT / ".venv-diffusers-original"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", baseline["python"], str(environment)], check=True)
    constraints = environment / "torch_constraints.txt"
    constraints.write_text("".join(f"{key}=={baseline[key]}\n" for key in ("torch", "torchvision")), encoding="utf-8")
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    f"torch=={baseline['torch']}", f"torchvision=={baseline['torchvision']}",
                    "--torch-backend", baseline["torch_backend"]], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    "-r", str(ROOT / "diffusers/original_requirements.txt"),
                    "-c", str(constraints)], check=True)
    check = "import sys,torch,torchvision; print(sys.version.split()[0]); print(torch.__version__); print(torchvision.__version__)"
    actual = subprocess.check_output([str(python), "-c", check], text=True).splitlines()
    expected = [baseline[key] for key in ("python", "torch", "torchvision")]
    if actual != expected:
        raise RuntimeError(f"환경 버전 불일치: {actual}, 기준: {expected}")
    print("실행 Python:", python)
    print("Python / torch / torchvision:", actual)


if __name__ == "__main__":
    setup()
