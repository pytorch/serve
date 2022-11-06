from examples.image_classifier.mnist.mnist_handler import MNISTDigitClassifier
from ts.utils.redis_cache import handler_cache


class MNISTDigitClassifierCached(MNISTDigitClassifier):
    def __init__(self):
        super(MNISTDigitClassifierCached, self).__init__()
        self.handle = handler_cache(host="localhost", port=6379, db=0, maxsize=2)(
            self.handle
        )
