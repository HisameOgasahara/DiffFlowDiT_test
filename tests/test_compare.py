import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.compare import compare_runs

class CompareTests(unittest.TestCase):
    def test_mismatched_experiments_and_shapes_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [root / name for name in ("left", "right")]
            for i, run in enumerate(runs):
                run.mkdir()
                (run / "generation.json").write_text(json.dumps({"seed": i}))
                (run / "models.json").write_text(json.dumps({"weights": {"dit": "same"}}))
                metrics = {"mode": "native", "status": "complete", "effective_sampler": "euler", "stages": {}}
                (run / "metrics.json").write_text(json.dumps(metrics))
                torch.save(torch.zeros(1, i + 1), run / "condition_prompt.pt")
                torch.save(torch.tensor([float('nan')]), run / "bad.pt")
                Image.new("RGB", (16, 16), color=(i * 255, 0, 0)).save(run / "image.png")
            compare_runs(runs, root / "comparison")
            result = json.loads((root / "comparison/comparison.json").read_text())[0]
            self.assertFalse(result["same_generation"])
            self.assertEqual(result["tensors"]["condition_prompt.pt"]["status"], "shape_mismatch")
            self.assertEqual(result["tensors"]["bad.pt"]["status"], "nonfinite")
            self.assertGreater(result["image_rmse_0_255"], 0)
            self.assertTrue((root / "comparison/comparison.png").is_file())

if __name__ == "__main__":
    unittest.main()
