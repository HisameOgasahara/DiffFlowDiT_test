import ast
import importlib.util
import sys
import types
import unittest
from functools import wraps
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.runtime import make_comfy_sigmas, make_noise, add_source

def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace

class SchedulerTests(unittest.TestCase):
    def test_common_noise_matches_original_comfy_rng(self):
        hp = dict(seed=1113280783077040, height=832, width=1216)
        noise = make_noise(hp)
        original = load_functions(ROOT / "comfyui/upstream/comfy/sample.py", {"prepare_noise_inner"}, {"torch": torch})
        expected = original["prepare_noise_inner"](torch.zeros_like(noise), torch.manual_seed(hp["seed"]), None)
        self.assertTrue(torch.equal(noise, expected))

    def test_original_schedulers_share_euler_trajectory(self):
        add_source("diffusers")
        from diffusers import FlowMatchEulerDiscreteScheduler
        spec = importlib.util.spec_from_file_location("original_flow", ROOT / "diffsynth/upstream/diffsynth/diffusion/flow_match.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grid = make_comfy_sigmas(7, 3.0, 1000)
        comfy = load_functions(ROOT / "comfyui/upstream/comfy/k_diffusion/sampling.py", {"sample_euler", "to_d"},
                               {"torch": torch, "trange": lambda n, **kwargs: range(n),
                                "utils": types.SimpleNamespace(append_dims=lambda x, ndim: x[(...,) + (None,) * (ndim - x.ndim)])})
        start = torch.tensor([[[[0.5, -0.25]]]], dtype=torch.float32)
        constant_velocity = torch.full_like(start, 0.125)
        def denoiser(x, sigma, **kwargs):
            return x - sigma.reshape(-1, 1, 1, 1) * constant_velocity
        a = comfy["sample_euler"](denoiser, start.clone(), grid)
        hf = FlowMatchEulerDiscreteScheduler(shift=3)
        hf.set_timesteps(7)
        hf.sigmas, hf.timesteps = grid.clone(), grid[:-1] * 1000
        ds = module.FlowMatchScheduler("Z-Image")
        ds.set_timesteps(7)
        ds.sigmas, ds.timesteps = grid[:-1].clone(), grid[:-1] * 1000
        b, c = start.clone(), start.clone()
        for t in hf.timesteps:
            b = hf.step(constant_velocity, t, b, return_dict=False)[0]
            c = ds.step(constant_velocity, t, c)
        expected = start - grid[0] * constant_velocity
        for result in (a, b, c):
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_hf_custom_schedule_preserves_pipeline_signature(self):
        add_source("diffusers")
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.modular_pipelines.anima.before_denoise import retrieve_timesteps
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
        original = scheduler.set_timesteps
        grid = make_comfy_sigmas(4, 3, 1000)
        @wraps(original)
        def traced(*args, **kwargs):
            original(*args, **kwargs)
            scheduler.sigmas, scheduler.timesteps = grid, grid[:-1] * 1000
        scheduler.set_timesteps = traced
        timesteps, count = retrieve_timesteps(scheduler, device="cpu", sigmas=[1.0, 0.75, 0.5, 0.25])
        self.assertEqual(count, 4)
        torch.testing.assert_close(timesteps, grid[:-1] * 1000)

if __name__ == "__main__":
    unittest.main()
