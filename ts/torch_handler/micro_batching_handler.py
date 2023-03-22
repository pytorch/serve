from ts.torch_handler.image_classifier import ImageClassifier
from ts.torch_handler.micro_batching import MicroBatching


class MicroBatchingHandler(ImageClassifier):
    def __init__(self):
        mb_handle = MicroBatching()
        self.handle = mb_handle
