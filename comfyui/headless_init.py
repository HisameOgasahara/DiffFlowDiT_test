# Adapted from ComfyUI main.py, commit f427c3a285502cc452bbf134b810668844b79ffa.
# Copyright ComfyUI contributors. GPL-3.0-or-later; see upstream/LICENSE.
# Only allocator and DynamicVRAM initialization are retained; no server imports.
import logging

def initialize_before_torch():
    from comfy.cli_args import args, enables_dynamic_vram
    import cuda_malloc
    import comfy_aimdo.control
    if enables_dynamic_vram():
        headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
        comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)

def initialize_devices():
    from comfy.cli_args import args, enables_dynamic_vram
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.memory_management as memory
    import comfy_aimdo.control
    supported = mm.is_nvidia() or (mm.is_amd() and mm.rocm_version >= (7, 14))
    if args.enable_dynamic_vram or (enables_dynamic_vram() and supported):
        if not args.enable_dynamic_vram and mm.torch_version_numeric < (2, 8):
            logging.warning("DynamicVRAM requires PyTorch >= 2.8; legacy ModelPatcher is active.")
        elif comfy_aimdo.control.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices()):
            comfy_aimdo.control.set_log_info()
            mp.CoreModelPatcher = mp.ModelPatcherDynamic
            memory.aimdo_enabled = True
            logging.info("DynamicVRAM enabled (headless).")
    return memory.aimdo_enabled
