import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

def parse_run(backend):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/generation.json")
    parser.add_argument("--runtime", type=Path, default=ROOT / "config/runtime.json")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "matched_euler"], default="native")
    parser.add_argument("--trace", choices=["selected", "none"], default="selected")
    args = parser.parse_args()
    args.hp, args.options = read_json(args.config), read_json(args.runtime)
    if args.hp["batch_size"] != 1 or args.hp["denoise"] != 1:
        raise ValueError("첫 비교는 batch_size=1, denoise=1인 txt2img로 제한합니다.")
    if min(args.hp["width"], args.hp["height"]) < 16 or args.hp["width"] % 16 or args.hp["height"] % 16:
        raise ValueError("해상도는 양수이며 16의 배수여야 합니다.")
    if args.hp["steps"] < 1:
        raise ValueError("steps는 1 이상이어야 합니다.")
    args.backend = backend
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.models = args.models.resolve()
    args.weights = {key: args.models / "weights" / Path(value).name for key, value in args.hp["models"].items()}
    for path in args.weights.values():
        if not path.is_file():
            raise FileNotFoundError(f"모델 준비 셀을 먼저 실행하세요: {path}")
    manifest = read_json(args.models / "model_manifest.json")
    for key, path in args.weights.items():
        expected = manifest["weights"][key]
        if path.name != expected["filename"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"모델 파일이 준비 시점과 다릅니다: {path}")
    write_json(args.output / "generation.json", args.hp)
    write_json(args.output / "runtime.json", args.options)
    write_json(args.output / "models.json", manifest)
    return args

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def add_source(backend):
    path = ROOT / backend / "upstream"
    if backend == "diffusers":
        path /= "src"
    sys.path.insert(0, str(path))

def make_noise(hp):
    import torch
    return torch.randn((1, 16, hp["height"] // 8, hp["width"] // 8),
                       generator=torch.Generator(device="cpu").manual_seed(hp["seed"]), dtype=torch.float32)

def make_comfy_sigmas(steps, shift, training_steps):
    import torch
    t = torch.arange(1, training_steps + 1, dtype=torch.float32) / training_steps
    grid = shift * t / (1 + (shift - 1) * t)
    indices = [-(1 + int(i * len(grid) / steps)) for i in range(steps)]
    return torch.cat([grid[indices], torch.zeros(1)])

class Recorder:
    def __init__(self, args):
        import torch
        import psutil
        self.args, self.torch = args, torch
        self.selected = {0, 1, args.hp["steps"] - 1}
        self.data = {"backend": args.backend, "mode": args.mode, "trace": args.trace, "stages": {}, "tensor_metadata": {}, "status": "running"}
        self.data["source_commit"] = read_json(ROOT / "source_manifest.json")[args.backend]["commit"]
        self.data["environment"] = {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_capacity_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        }
        self.handles = []
        self.stage_stack = []
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        def poll():
            while not self.stop.wait(0.1):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()

    def save(self, name, tensor, step=None):
        if tensor is None or not self.torch.is_tensor(tensor):
            return
        if step is not None and step not in self.selected:
            return
        name = name if step is None else f"{name}_{step:03d}"
        self.data["tensor_metadata"][name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}
        if self.args.trace != "none":
            self.torch.save(tensor.detach().cpu().contiguous(), self.args.output / f"{name}.pt")

    def observe_model(self, name, model):
        self.data.setdefault("models", {})[name] = {
            "class": type(model).__name__,
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
        }
        counts = {"calls": 0}
        self.data.setdefault("forward_calls", {})[name] = counts
        def hook(module, inputs, kwargs, output):
            counts["calls"] += 1
            if counts["calls"] == 1:
                for key, value in kwargs.items():
                    self.save(f"{name}_input_{key}", value)
                for index, value in enumerate(inputs):
                    self.save(f"{name}_input_{index}", value)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                self.save(f"{name}_first_output", output)
        self.handles.append(model.register_forward_hook(hook, with_kwargs=True))

    @contextlib.contextmanager
    def stage(self, name):
        torch = self.torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if self.stage_stack:
                parent = self.stage_stack[-1]
                parent[0] = max(parent[0], torch.cuda.max_memory_allocated())
                parent[1] = max(parent[1], torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        peak = [0, 0]
        self.stage_stack.append(peak)
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak[0] = max(peak[0], torch.cuda.max_memory_allocated())
                peak[1] = max(peak[1], torch.cuda.max_memory_reserved())
            self.stage_stack.pop()
            if self.stage_stack:
                parent = self.stage_stack[-1]
                parent[0] = max(parent[0], peak[0])
                parent[1] = max(parent[1], peak[1])
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.data["stages"][name] = {
                "seconds": time.perf_counter() - start,
                "peak_allocated_bytes": peak[0],
                "peak_reserved_bytes": peak[1],
            }
            write_json(self.args.output / "metrics.json", self.data)

    def finish(self, error=None):
        self.stop.set()
        self.thread.join()
        for handle in self.handles:
            handle.remove()
        self.data.update(status="failed" if error else "complete", error=str(error) if error else None,
                         total_seconds=time.perf_counter() - self.started, peak_cpu_rss_bytes=self.peak_rss)
        write_json(self.args.output / "metrics.json", self.data)

def save_image(tensor, path, channel_last=False):
    import numpy as np
    from PIL import Image
    if tensor.ndim == 4:
        tensor = tensor[0]
    if not channel_last:
        tensor = tensor.permute(1, 2, 0)
    Image.fromarray(np.clip(tensor.detach().float().cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(path)
