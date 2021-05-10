import importlib
import os
import torch
from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import list_classes_from_module


class VGGImageClassifier(ImageClassifier):
    """
    Overriding the model loading code as a workaround for issue :
    https://github.com/pytorch/serve/issues/535
    https://github.com/pytorch/vision/issues/2473
    """

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError("Expected only one class as model definition. {}".format(
                model_class_definitions))

        model_class = model_class_definitions[0]
        state_dict = torch.load(model_pt_path)
        model = model_class()
        model.load_state_dict(state_dict)
        return model

