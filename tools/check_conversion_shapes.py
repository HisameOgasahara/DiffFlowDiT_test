"""Validate official HF key conversion using checkpoint headers, without weights."""
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, add_source
add_source("diffusers")
sys.path.insert(0, str(ROOT / "diffusers/upstream/scripts"))
import torch
import convert_anima_to_diffusers as converter

def state(name):
    header = json.loads((ROOT / f"validation/model_headers/{name}.json").read_text())
    return {key: torch.empty(value["shape"], device="meta", dtype=torch.float16)
            for key, value in header.items() if key != "__metadata__"}

dit, conditioner = converter.split_anima_transformer_checkpoint(state("dit"))
transformer = converter.convert_transformer("Cosmos-2.0-Diffusion-2B-Text2Image", state_dict=dit, weights_only=True)
conditioner = converter.convert_text_conditioner(conditioner)
encoder = converter.convert_text_encoder(state("text_encoder"))
vae = converter.convert_qwen_image_vae(state("vae"))
print("PASS: three checkpoint headers map strictly to HF transformer, text conditioner, Qwen3 and VAE.")
print({name: sum(p.numel() for p in module.parameters()) for name, module in
       (("transformer", transformer), ("text_conditioner", conditioner), ("text_encoder", encoder), ("vae", vae))})
