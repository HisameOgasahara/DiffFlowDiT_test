import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.extract_png import read_metadata, extract_config

class BundleTests(unittest.TestCase):
    def test_png_matches_saved_generation(self):
        extracted = extract_config(read_metadata(ROOT / "config/reference.png"))
        saved = json.loads((ROOT / "config/generation.json").read_text(encoding="utf-8"))
        self.assertEqual(extracted, saved)
        self.assertEqual(saved["sampler_name"], "er_sde")
        self.assertEqual(saved["seed"], 1113280783077040)
        self.assertIn("plana (blue archive)", saved["prompt"])
        self.assertFalse(saved["lora_applied"])

    def test_upstream_files_unchanged(self):
        manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        for backend, record in manifest.items():
            for relative, expected in record["files_sha256"].items():
                path = ROOT / backend / "upstream" / relative
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, str(path))

    def test_notebooks_compile_and_have_no_outputs(self):
        notebooks = list(ROOT.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["metadata"]["colab"]["gpuType"], "T4")
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile(cell["source"], f"{path.name}:{index}", "exec")
                    self.assertEqual(cell["outputs"], [])

    def test_runner_does_not_import_web_ui(self):
        tree = ast.parse((ROOT / "comfyui/run.py").read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        self.assertFalse({"nodes", "server", "execution", "main"} & set(modules))

    def test_checkpoint_headers_match_diffsynth_registry(self):
        registry = (ROOT / "diffsynth/upstream/diffsynth/configs/model_configs.py").read_text(encoding="utf-8")
        for path in (ROOT / "validation/model_headers").glob("*.json"):
            header = json.loads(path.read_text(encoding="utf-8"))
            keys = []
            for key, value in header.items():
                if key != "__metadata__":
                    keys.extend([key + ":" + "_".join(map(str, value["shape"])), key])
            digest = hashlib.md5(",".join(sorted(keys)).encode()).hexdigest()
            self.assertIn(digest, registry, path.name)

if __name__ == "__main__":
    unittest.main()
