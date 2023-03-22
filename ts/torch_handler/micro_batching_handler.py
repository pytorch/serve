import json
import logging
from pathlib import Path

from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.micro_batching import MicroBatching

logger = logging.getLogger(__name__)


class MicroBatchingHandler(ImageClassifier):
    def __init__(self):
        mb_handle = MicroBatching(self)
        self.handle = mb_handle

    def initialize(self, ctx):
        super().initialize(ctx)

        config_file = Path(ctx.system_properties["model_dir"], "micro_batching.json")
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)

                self.handle.paralellism = config["parallelism"]
                self.handle.micro_batch_size = config["micro_batch_size"]
                logger.info(f"Successfully loaded {config_file}")
            except KeyError as e:
                logger.error(
                    f"Failed to load micro batching configuration! Keys missing:{e}"
                )
