import logging
import os

import torch
from torch._dynamo.utils import counters

from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)


class TorchInductorCacheHandler(ImageClassifier):
    """
    Diffusion-Fast handler class for text to image generation.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.context = ctx
        self.manifest = ctx.manifest
        properties = ctx.system_properties

        if (
            "handler" in ctx.model_yaml_config
            and "torch_inductor_caching" in ctx.model_yaml_config["handler"]
        ):
            if ctx.model_yaml_config["handler"]["torch_inductor_caching"].get(
                "torch_inductor_fx_graph_cache", False
            ):
                torch._inductor.config.fx_graph_cache = True
            if (
                "torch_inductor_cache_dir"
                in ctx.model_yaml_config["handler"]["torch_inductor_caching"]
            ):
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = ctx.model_yaml_config[
                    "handler"
                ]["torch_inductor_caching"]["torch_inductor_cache_dir"]

        super().initialize(ctx)
        self.initialized = True

    def inference(self, data, *args, **kwargs):
        with torch.inference_mode():
            marshalled_data = data.to(self.device)
            results = self.model(marshalled_data, *args, **kwargs)

        # Debugs for FX Graph Cache hit
        if torch._inductor.config.fx_graph_cache:
            fx_graph_cache_hit, fx_graph_cache_miss = (
                counters["inductor"]["fxgraph_cache_hit"],
                counters["inductor"]["fxgraph_cache_miss"],
            )
            logger.info(
                f'TorchInductor FX Graph cache hit {counters["inductor"]["fxgraph_cache_hit"]}, FX Graph cache miss {counters["inductor"]["fxgraph_cache_miss"]}'
            )
        return results
