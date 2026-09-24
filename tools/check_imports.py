import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import add_source

parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=["comfyui", "diffusers", "diffsynth"])
args = parser.parse_args()
add_source(args.backend)
if args.backend == "comfyui":
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.sd
    import comfy.sample
    print("ComfyUI core:", comfy.sd.__file__)
elif args.backend == "diffusers":
    from diffusers import AnimaAutoBlocks, CosmosTransformer3DModel, AnimaTextConditioner, AutoencoderKLQwenImage
    pipe = AnimaAutoBlocks().init_pipeline()
    print("Diffusers blocks:", type(pipe).__name__)
elif args.backend == "diffsynth":
    from diffsynth.pipelines.anima_image import AnimaImagePipeline
    import torch
    pipe = AnimaImagePipeline(device="cpu", torch_dtype=torch.float32)
    print("DiffSynth units:", [type(unit).__name__ for unit in pipe.units])
