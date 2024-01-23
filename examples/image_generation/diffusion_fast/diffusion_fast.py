import logging

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler
from pipeline_utils import conv_filter_fn, dynamic_quant_filter_fn
from torchao.quantization import (
    apply_dynamic_quant,
    change_linear_weights_to_int4_woqtensors,
    change_linear_weights_to_int8_woqtensors,
    swap_conv2d_1x1_to_linear,
)

logger = logging.getLogger(__name__)


class DiffusionFast:
    def __init__(
        self,
        do_quant: str = None,
        compile_unet: bool = False,
        compile_vae: bool = False,
        ckpt: str = "stabilityai/stable-diffusion-xl-base-1.0",
        upcast_vae: bool = False,
        enable_fused_projections: bool = False,
        no_sdpa: bool = False,
        compile_mode: str = None,
        no_bf16: bool = False,
        change_comp_config: bool = False,
    ) -> None:
        self.do_quant = do_quant
        self.compile_unet = compile_unet
        self.compile_vae = compile_vae
        self.ckpt = ckpt
        self.upcast_vae = upcast_vae
        self.enable_fused_projections = enable_fused_projections
        self.no_sdpa = no_sdpa
        self.compile_mode = compile_mode
        self.no_bf16 = no_bf16
        self.change_comp_config = change_comp_config

        if self.do_quant and not self.compile_unet:
            raise ValueError("Compilation for UNet must be enabled when quantizing.")
        if self.do_quant and not self.compile_vae:
            raise ValueError("Compilation for VAE must be enabled when quantizing.")

        dtype = torch.float32 if self.no_bf16 else torch.bfloat16
        logger.info(f"Using dtype: {dtype}")

        if self.ckpt != "runwayml/stable-diffusion-v1-5":
            self.pipe = DiffusionPipeline.from_pretrained(
                self.ckpt, torch_dtype=dtype, use_safetensors=True
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.ckpt, torch_dtype=dtype, use_safetensors=True, safety_checker=None
            )
            # As the default scheduler of SD v1-5 doesn't have sigmas device placement
            # (https://github.com/huggingface/diffusers/pull/6173)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

        if not self.upcast_vae and self.ckpt != "runwayml/stable-diffusion-v1-5":
            logger.info("Using a more numerically stable VAE.")
            self.pipe.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
            )

        if self.enable_fused_projections:
            logger.info("Enabling fused QKV projections for both UNet and VAE.")
            self.pipe.fuse_qkv_projections()

        if self.upcast_vae and self.ckpt != "runwayml/stable-diffusion-v1-5":
            logger.info("Upcasting VAE.")
            self.pipe.upcast_vae()

        if self.no_sdpa:
            logger.info("Using vanilla attention.")
            self.pipe.unet.set_default_attn_processor()
            self.pipe.vae.set_default_attn_processor()

        self.pipe = self.pipe.to("cuda")

        if self.compile_unet:
            self.pipe.unet.to(memory_format=torch.channels_last)
            logger.info("Compile UNet.")
            swap_conv2d_1x1_to_linear(self.pipe.unet, conv_filter_fn)
            if self.compile_mode == "max-autotune" and self.change_comp_config:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True

            if self.do_quant:
                logger.info("Apply quantization to UNet.")
                if self.do_quant == "int4weightonly":
                    change_linear_weights_to_int4_woqtensors(self.pipe.unet)
                elif self.do_quant == "int8weightonly":
                    change_linear_weights_to_int8_woqtensors(self.pipe.unet)
                elif self.do_quant == "int8dynamic":
                    apply_dynamic_quant(self.pipe.unet, dynamic_quant_filter_fn)
                else:
                    raise ValueError(f"Unknown do_quant value: {self.do_quant}.")
                torch._inductor.config.force_fuse_int_mm_with_mul = True
                torch._inductor.config.use_mixed_mm = True

            self.pipe.unet = torch.compile(
                self.pipe.unet, mode=self.compile_mode, fullgraph=True
            )

        if self.compile_vae:
            self.pipe.vae.to(memory_format=torch.channels_last)
            logger.info("Compile VAE.")
            swap_conv2d_1x1_to_linear(self.pipe.vae, conv_filter_fn)

            if self.compile_mode == "max-autotune" and self.change_comp_config:
                torch._inductor.config.conv_1x1_as_mm = True
                torch._inductor.config.coordinate_descent_tuning = True
                torch._inductor.config.epilogue_fusion = False
                torch._inductor.config.coordinate_descent_check_all_directions = True

            if self.do_quant:
                logger.info("Apply quantization to VAE.")
                if self.do_quant == "int4weightonly":
                    change_linear_weights_to_int4_woqtensors(self.pipe.vae)
                elif self.do_quant == "int8weightonly":
                    change_linear_weights_to_int8_woqtensors(self.pipe.vae)
                elif self.do_quant == "int8dynamic":
                    apply_dynamic_quant(self.pipe.vae, dynamic_quant_filter_fn)
                else:
                    raise ValueError(f"Unknown do_quant value: {self.do_quant}.")
                torch._inductor.config.force_fuse_int_mm_with_mul = True
                torch._inductor.config.use_mixed_mm = True

            self.pipe.vae.decode = torch.compile(
                self.pipe.vae.decode, mode=self.compile_mode, fullgraph=True
            )

        self.pipe.set_progress_bar_config(disable=True)
