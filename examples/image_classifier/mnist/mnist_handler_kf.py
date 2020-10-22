import logging
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
import torch

logger = logging.getLogger(__name__)


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            if isinstance(row, dict):
                image = row.get("data") or row.get("body") or row
            else:
                image = row
            logger.info("Mnist image code %s", image)
            image = torch.FloatTensor(image)
            logger.info("Mnist image code tensor %s", image)
            images.append(image)

        return torch.stack(images)

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
            output_explain (list): Defaults to None if no predict request is made,
            otherwise list of explanations are passed.

        Returns:
            list : A list of dictionary with predictons and explanations are returned.
        """
        response = {}
        response["predictions"] = data.argmax(1).tolist()
        return [response]
