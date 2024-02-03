import logging
import os

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
            if (
                "torch_inductor_fx_graph_cache"
                in ctx.model_yaml_config["handler"]["torch_inductor_caching"]
            ):
                os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = ctx.model_yaml_config[
                    "handler"
                ]["torch_inductor_caching"]["torch_inductor_fx_graph_cache"]
            if (
                "torch_inductor_cache_dir"
                in ctx.model_yaml_config["handler"]["torch_inductor_caching"]
            ):
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = ctx.model_yaml_config[
                    "handler"
                ]["torch_inductor_caching"]["torch_inductor_cache_dir"]

        super().initialize(ctx)
        self.initialized = True
