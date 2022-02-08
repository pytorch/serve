from ts.torch_handler.image_classifier import ImageClassifier


class ResnetHandler(ImageClassifier):
    def __init__(self) -> None:
        super().__init__()