"""Run the upstream converters one component per process to bound CPU RAM."""
import argparse
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT, add_source, read_json, write_json

def convert(models, component, dtype_name):
    add_source("diffusers")
    sys.path.insert(0, str(ROOT / "diffusers/upstream/scripts"))
    import torch
    from safetensors.torch import load_file
    import convert_anima_to_diffusers as original
    manifest = read_json(models / "model_manifest.json")
    weights = models / "weights"
    dtype = getattr(torch, dtype_name)
    if component in ("transformer", "text_conditioner"):
        state = load_file(str(weights / manifest["weights"]["dit"]["filename"]))
        transformer_state, conditioner_state = original.split_anima_transformer_checkpoint(state)
        del state
        if component == "transformer":
            del conditioner_state
            model = original.convert_transformer("Cosmos-2.0-Diffusion-2B-Text2Image", state_dict=transformer_state, weights_only=True)
            del transformer_state
        else:
            del transformer_state
            model = original.convert_text_conditioner(conditioner_state)
            del conditioner_state
    elif component == "text_encoder":
        state = load_file(str(weights / manifest["weights"][component]["filename"]))
        model = original.convert_text_encoder(state)
        del state
    elif component == "vae":
        state = load_file(str(weights / manifest["weights"][component]["filename"]))
        model = original.convert_qwen_image_vae(state)
        del state
    else:
        raise ValueError(component)
    gc.collect()
    model.to(dtype=dtype)
    destination = models / "diffusers_converted" / dtype_name / component
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="2GB")
    write_json(destination / "conversion_source.json", {"weights": manifest["weights"], "dtype": dtype_name,
               "source_commit": read_json(ROOT / "source_manifest.json")["diffusers"]["commit"]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--component", choices=["transformer", "text_conditioner", "text_encoder", "vae"], required=True)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    args = parser.parse_args()
    convert(args.models, args.component, args.dtype)
