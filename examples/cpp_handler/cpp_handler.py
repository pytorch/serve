"""
A handler to load TorchScript model & do inference using using C++ API
"""
import logging
import os
import io
import pathlib
from torch.utils.cpp_extension import load
import torch
from PIL import Image
from torchvision import transforms
from ts.utils.util import map_class_to_label, load_label_mapping

logger = logging.getLogger(__name__)


class CPPHandler(object):
    """
    A handler to load TorchScript model & do inference using using C++ API defined in
    torch_cpp_python_bindings.cpp
    """

    topk = 5

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.torch_cpp_python_module = None

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ":" + str(properties.get("gpu_id"))
                                   if torch.cuda.is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # model def file
        model_file = self.manifest['model'].get('modelFile', '')
        self.torch_api_type = properties["torch_api_type"]

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)

        if model_file:
            raise Exception("Eager models are not supported using CPP API")

        logger.info('C++ Handler: Loading Torch Script model using CPP API')
        source_path = pathlib.Path(__file__).parent.absolute()
        cpp_source_path = os.path.join(source_path, "torch_cpp_python_bindings.cpp")
        self.torch_cpp_python_module = load(name="torch_cpp_python_bindings",
                                            sources=[cpp_source_path],
                                            verbose=True,
                                            extra_ldflags=['-lopencv_core', '-lopencv_imgcodecs']
                                            )
        self.model = self.torch_cpp_python_module.initialize(model_pt_path, self.map_location,
                                                             str(self.device))

        logger.debug('Model file %s loaded successfully', model_pt_path)

        self.initialized = True

    def handle(self, data, context):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        self.context = context

        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            images.append(image)
        probs, classes = self.torch_cpp_python_module.handle(self.model, bytes(images[0]), str(self.device), self.topk)
        probs, classes = probs.tolist(), classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)


if __name__ == "__main__":
    import ts
    import numpy as np
    import pathlib
    from ts.context import Context

    source_path = pathlib.Path(__file__).parent.absolute()
    model_dir = "/Users/mahesh/tmp/dense_cpp"
    m_path = 'densenet161.pt'
    context = Context('tf_model', model_dir, {'model': {'serializedFile': m_path,
                                                        'modelFile': None}},
                      1, 'cpu', ts.__version__, "python")

    data = np.random.uniform(size=(3, 3, 224, 224)).astype('float32')

    f = open("/Users/mahesh/git/serve/examples/image_classifier/kitten.jpg", "rb")
    data = [{"body": f.read()}]

    ic = CPPHandler()
    ic.initialize(context)
    print(ic.handle(data, context))
