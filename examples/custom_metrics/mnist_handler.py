import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """
        metrics = self.context.metrics
        input = data[0].get('body')
        metrics.add_size('SizeOfImage', len(input) / 1024, None, 'kB')
        return ImageClassifier.preprocess(self, data)

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionary with predictons and explanations are returned.
        """
        return data.argmax(1).flatten().tolist()
