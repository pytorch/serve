import io

from PIL import Image
from torchvision import transforms

from ts.torch_handler.image_classifier import ImageClassifier


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here methods preprocess() and postprocess() have been overridden while others are reused from parent class.
    """

    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns a tensor.
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = Image.open(io.BytesIO(image))
        image = mnist_transform(image)

        # Convert 2D image to 1D vector
        image = image.unsqueeze(0)

        return image

    def postprocess(self, inference_output):
        _, y_hat = inference_output.max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]
