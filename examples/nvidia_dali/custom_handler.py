# pylint: disable=W0223
# Details : https://github.com/PyCQA/pylint/issues/3098
"""
Base module for all vision handlers
"""
from torch.profiler import ProfilerActivity

from ts.torch_handler.dali_image_classifier import DALIImageClassifier


class DALIMNISTDigitClassifier(DALIImageClassifier):
    """
    Base class for all vision handlers
    """

    def __init__(self):
        super(DALIMNISTDigitClassifier, self).__init__()

    def initialize(self, context):
        super().initialize(context)
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        """
        return data.argmax(1).tolist()
