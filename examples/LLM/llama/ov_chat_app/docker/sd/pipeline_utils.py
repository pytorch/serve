import torch
import openvino.torch
import logging

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)

logger = logging.getLogger(__name__)

def load_pipeline(
    ckpt: str,
    compile_unet: bool,
    compile_vae: bool,
    compile_mode: str,
    change_comp_config: bool,
    compile_options: str,
    is_xl: bool,
    is_lcm: bool
):
    """Loads the SDXL pipeline."""

    dtype = torch.float16
    logger.info(f"Using dtype: {dtype}")
    compile_options_str = ", ".join([f"{k} {v}" for k, v in compile_options.items()])
    logger.info(f"Compiled model with {compile_options_str}")

    if is_lcm:
        unet = UNet2DConditionModel.from_pretrained(f"{ckpt}/lcm/", torch_dtype=dtype)
        pipe = DiffusionPipeline.from_pretrained(ckpt, unet=unet, torch_dtype=dtype, variant="fp16")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_options)
        
    elif is_xl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
                    ckpt, torch_dtype=dtype, use_safetensors=True
                )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            ckpt, torch_dtype=dtype, use_safetensors=True, safety_checker=None
        )

    if compile_unet:
        print("Compile UNet.")
        if compile_mode == "max-autotune" and change_comp_config:
            pipe.unet.to(memory_format=torch.channels_last)
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        pipe.unet = torch.compile(pipe.unet, **compile_options)

    if compile_vae:
        print("Compile VAE.")
        if compile_mode == "max-autotune" and change_comp_config:
            pipe.vae.to(memory_format=torch.channels_last)
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

        pipe.vae.decode = torch.compile(pipe.vae.decode, **compile_options)
    logger.info(f"Compiled model with {compile_options_str}")

    pipe.set_progress_bar_config(disable=True)
    return pipe
