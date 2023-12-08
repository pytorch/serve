import logging

from ts.handler_utils.micro_batching import MicroBatching
from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)


class MicroBatchingHandler(ImageClassifier):
    def __init__(self):
        mb_handle = MicroBatching(self)
        self.handle = mb_handle

    def initialize(self, ctx):
        super().initialize(ctx)

        parallelism = ctx.model_yaml_config.get("micro_batching", {}).get(
            "parallelism", None
        )
        if parallelism:
            logger.info(
                f"Setting micro batching parallelism  from model_config_yaml: {parallelism}"
            )
            self.handle.parallelism = parallelism

        micro_batch_size = ctx.model_yaml_config.get("micro_batching", {}).get(
            "micro_batch_size", 1
        )
        logger.info(f"Setting micro batching size: {micro_batch_size}")
        self.handle.micro_batch_size = micro_batch_size
