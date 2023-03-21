import importlib
import logging
import os
from abc import ABC

import torch
from captum.attr import IntegratedGradients
from PIL import Image

from ..utils.util import (
    list_classes_from_module,
    load_compiler_config,
    load_label_mapping,
)

logger = logging.getLogger(__name__)

# Possible values for backend in utils.py
def check_pt2_enabled():
    try:
        import torch._dynamo

        pt2_enabled = True
        if torch.cuda.is_available():
            # If Ampere enable tensor cores which will give better performance
            # Ideally get yourself an A10G or A100 for optimal performance
            if torch.cuda.get_device_capability() >= (8, 0):
                torch.backends.cuda.matmul.allow_tf32 = True
    except ImportError as error:
        logger.warning(
            "dynamo/inductor are not installed. \n For GPU please run pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117 \n for CPU please run pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
        )
        pt2_enabled = False
    return pt2_enabled


def vision_initialize(obj, context):
    obj.ig = IntegratedGradients(obj.model)
    obj.initialized = True
    properties = context.system_properties
    if not properties.get("limit_max_image_pixels"):
        Image.MAX_IMAGE_PIXELS = None


def _load_pickled_model(obj, model_dir, model_file, model_pt_path):
    """
    Loads the pickle file from the given model path.

    Args:
        model_dir (str): Points to the location of the model artefacts.
        model_file (.py): the file which contains the model class.
        model_pt_path (str): points to the location of the model pickle file.

    Raises:
        RuntimeError: It raises this error when the model.py file is missing.
        ValueError: Raises value error when there is more than one class in the label,
                    since the mapping supports only one label per class.

    Returns:
        serialized model file: Returns the pickled pytorch model file
    """
    model_def_path = os.path.join(model_dir, model_file)
    if not os.path.isfile(model_def_path):
        raise RuntimeError("Missing the model.py file")

    module = importlib.import_module(model_file.split(".")[0])
    model_class_definitions = list_classes_from_module(module)
    if len(model_class_definitions) != 1:
        raise ValueError(
            "Expected only one class as model definition. {}".format(
                model_class_definitions
            )
        )

    model_class = model_class_definitions[0]
    model = model_class()
    if model_pt_path:
        state_dict = torch.load(model_pt_path, map_location=obj.device)
        model.load_state_dict(state_dict)
    return model


def base_initialize(obj, context):
    """Initialize function loads the model.pt file and initialized the model object.
       First try to load torchscript else load eager mode state_dict based model.

    Args:
        context (context): It is a JSON Object containing information
        pertaining to the model artifacts parameters.

    Raises:
        RuntimeError: Raises the Runtime error when the model.py is missing

    """
    ipex_enabled = False
    if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
        try:
            import intel_extension_for_pytorch as ipex

            ipex_enabled = True
        except ImportError as error:
            logger.warning(
                "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
            )

    properties = context.system_properties
    obj.map_location = (
        "cuda"
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else "cpu"
    )
    obj.device = torch.device(
        obj.map_location + ":" + str(properties.get("gpu_id"))
        if torch.cuda.is_available() and properties.get("gpu_id") is not None
        else obj.map_location
    )
    obj.manifest = context.manifest

    model_dir = properties.get("model_dir")
    obj.model_pt_path = None
    if "serializedFile" in obj.manifest["model"]:
        serialized_file = obj.manifest["model"]["serializedFile"]
        obj.model_pt_path = os.path.join(model_dir, serialized_file)

    if obj.model_pt_path:
        if obj.model_pt_path.endswith("onnx"):
            try:
                # import numpy as np
                import onnxruntime as ort
                import psutil

                onnx_enabled = True
                logger.info("ONNX enabled")
            except ImportError as error:
                onnx_enabled = False
                logger.warning("proceeding without onnxruntime")

    # model def file
    model_file = obj.manifest["model"].get("modelFile", "")

    if model_file:
        logger.debug("Loading eager model")
        obj.model = _load_pickled_model(obj, model_dir, model_file, obj.model_pt_path)
        obj.model.to(obj.device)
        obj.model.eval()

    # Convert your model by following instructions: https://pytorch.org/tutorials/intermediate/nvfuser_intro_tutorial.html
    # For TensorRT support follow instructions here: https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html#getting-started-with-python-api
    elif obj.model_pt_path.endswith(".pt"):
        obj.model = obj._load_torchscript_model(obj.model_pt_path)
        obj.model.eval()

    # Convert your model by following instructions: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    # TODO(msaroufim): Refactor into utils https://github.com/pytorch/serve/issues/1631
    elif obj.model_pt_path.endswith(".onnx") and onnx_enabled:
        # obj.model = obj._load_onnx_model(obj.model_pt_path)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if obj.map_location == "cuda"
            else ["CPUExecutionProvider"]
        )

        # Set the right inference options, we can add more options here depending on what people want
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        # Start an inference session
        ort_session = ort.InferenceSession(
            obj.model_pt_path, providers=providers, sess_options=sess_options
        )
        obj.model = ort_session

    else:
        raise RuntimeError("No model weights could be loaded")

    optimization_config = os.path.join(model_dir, "compile.json")
    backend = load_compiler_config(optimization_config)

    # PT 2.0 support is opt in
    if check_pt2_enabled() and backend:
        # Compilation will delay your model initialization
        try:
            obj.model = torch.compile(
                obj.model, backend=backend, mode="reduce-overhead"
            )
            logger.info(f"Compiled model with backend {backend}")
        except:
            logger.warning(
                f"Compiling model model with backend {backend} has failed \n Proceeding without compilation"
            )

    elif ipex_enabled:
        obj.model = obj.model.to(memory_format=torch.channels_last)
        obj.model = ipex.optimize(obj.model)

    logger.debug("Model file %s loaded successfully", obj.model_pt_path)

    # Load class mapping for classifiers
    mapping_file_path = os.path.join(model_dir, "index_to_name.json")
    obj.mapping = load_label_mapping(mapping_file_path)

    obj.initialized = True


def vision_preprocess(obj, data):
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
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))
            image = obj.image_processing(image)
        else:
            # if the image is a list
            image = torch.FloatTensor(image)

        images.append(image)

    return torch.stack(images).to(obj.device)


class Caller(ABC):
    def __call__(self, *args):
        if self._prev:
            self._prev(*args)

        self._method(*args)


class VisionPreproc(Caller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = vision_preprocess


class BaseInit(Caller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_initialize


class VisionInit(Caller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = vision_initialize
