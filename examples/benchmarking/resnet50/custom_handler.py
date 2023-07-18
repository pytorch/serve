import io

import torch

from ts.handler_utils.timer import timed
from ts.torch_handler.image_classifier import ImageClassifier


class ResNet50Classifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    def __init__(self):
        super(ResNet50Classifier, self).__init__()

    @timed
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
            image = row.get("data") or row.get("body")
            image = torch.load(io.BytesIO(image))
            # if isinstance(image, str):
            #    # if the image is a string of bytesarray.
            #    image = base64.b64decode(image)

            ## If the image is sent as bytesarray
            # if isinstance(image, (bytearray, bytes)):
            #    image = Image.open(io.BytesIO(image))
            #    image = self.image_processing(image)
            # else:
            #    # if the image is a list
            #    image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    # def _load_torchscript_model(self, model_pt_path):
    #    """Loads the PyTorch model and returns the NN model object.

    #    Args:
    #        model_pt_path (str): denotes the path of the model file.

    #    Returns:
    #        (NN Model Object) : Loads the model object.
    #    """
    #    # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved
    #    model = torch.jit.load(model_pt_path)
    #    model.to(self.device)
    #    return model

    # def _load_pickled_model(self, model_dir, model_file, model_pt_path):
    #    model_def_path = os.path.join(model_dir, model_file)
    #    if not os.path.isfile(model_def_path):
    #        raise RuntimeError("Missing the model.py file")

    #    module = importlib.import_module(model_file.split(".")[0])
    #    model_class_definitions = list_classes_from_module(module)
    #    if len(model_class_definitions) != 1:
    #        raise ValueError("Expected only one class as model definition. {}".format(
    #            model_class_definitions))

    #    model_class = model_class_definitions[0]
    #    state_dict = torch.load(model_pt_path)
    #    model = model_class()
    #    model.load_state_dict(state_dict)
    #    return model
