import importlib
import logging
import os
import time

import psutil
import torch

from ts.handler_utils.caller import InitCaller, PipeCaller
from ts.utils.util import (
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


def _load_torchscript_model(obj, model_pt_path):
    """Loads the PyTorch model and returns the NN model object.
    Args:
        model_pt_path (str): denotes the path of the model file.
    Returns:
        (NN Model Object) : Loads the model object.
    """
    return torch.jit.load(model_pt_path, map_location=obj.device)


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
        obj.model = _load_torchscript_model(obj, obj.model_pt_path)
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


def base_inference(obj, data, *args, **kwargs):
    """
    The Inference Function is used to make a prediction call on the given input request.
    The user needs to override the inference function to customize it.
    Args:
        data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
        The shape should match the model input shape.
    Returns:
        Torch Tensor : The Predicted Torch Tensor is returned in this function.
    """
    with torch.no_grad():
        marshalled_data = data.to(obj.device)
        results = obj.model(marshalled_data, *args, **kwargs)
    return results


def base_preprocess(obj, data):
    """
    Preprocess function to convert the request input to a tensor(Torchserve supported format).
    The user needs to override to customize the pre-processing
    Args :
        data (list): List of the data from the request input.
    Returns:
        tensor: Returns the tensor data of the input
    """
    return torch.as_tensor(data, device=obj.device)


def base_postprocess(obj, data):
    """
    The post process function makes use of the output from the inference and converts into a
    Torchserve supported response output.
    Args:
        data (Torch Tensor): The torch tensor received from the prediction output of the model.
    Returns:
        List: The post process function returns a list of the predicted output.
    """

    return data.tolist()


def base_handle(obj, data, context):
    """Entry point for default handler. It takes the data from the input request and returns
        the predicted outcome for the input.
    Args:
        data (list): The input data that needs to be made a prediction request on.
        context (Context): It is a JSON Object containing information pertaining to
                            the model artefacts parameters.
    Returns:
        list : Returns a list of dictionary with the predicted response.
    """

    # It can be used for pre or post processing if needed as additional request
    # information is available in context
    start_time = time.time()

    obj.context = context
    metrics = obj.context.metrics

    is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
    if is_profiler_enabled:
        if PROFILER_AVAILABLE:
            output, _ = obj._infer_with_profiler(data=data)
        else:
            raise RuntimeError(
                "Profiler is enabled but current version of torch does not support."
                "Install torch>=1.8.1 to use profiler."
            )
    else:
        if _is_describe(obj):
            output = [describe_handle(obj)]
        else:
            data_preprocess = obj.preprocess(data)

            if not _is_explain(obj):
                output = obj.inference(data_preprocess)
                output = obj.postprocess(output)
            else:
                output = explain_handle(obj, data_preprocess, data)

    stop_time = time.time()
    metrics.add_time(
        "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
    )
    return output


def explain_handle(obj, data_preprocess, raw_data):
    """Captum explanations handler
    Args:
        data_preprocess (Torch Tensor): Preprocessed data to be used for captum
        raw_data (list): The unprocessed data to get target from the request
    Returns:
        dict : A dictionary response with the explanations response.
    """
    output_explain = None
    inputs = None
    target = 0

    logger.info("Calculating Explanations")
    row = raw_data[0]
    if isinstance(row, dict):
        logger.info("Getting data and target")
        inputs = row.get("data") or row.get("body")
        target = row.get("target")
        if not target:
            target = 0

    output_explain = obj.get_insights(data_preprocess, inputs, target)
    return output_explain


def _is_explain(obj):
    if obj.context and obj.context.get_request_header(0, "explain"):
        if obj.context.get_request_header(0, "explain") == "True":
            obj.explain = True
            return True
    return False


def _is_describe(obj):
    if obj.context and obj.context.get_request_header(0, "describe"):
        if obj.context.get_request_header(0, "describe") == "True":
            return True
    return False


def describe_handle(obj):
    """Customized describe handler
    Returns:
        dict : A dictionary response.
    """
    # pylint: disable=unnecessary-pass
    pass


def _infer_with_profiler(obj, data):
    """Custom method to generate pytorch profiler traces for preprocess/inference/postprocess
    Args:
        data (list): The input data that needs to be made a prediction request on.
    Returns:
        output : Returns a list of dictionary with the predicted response.
        prof: pytorch profiler object
    """
    # Setting the default profiler arguments to profile cpu, gpu usage and record shapes
    # User can override this argument based on the requirement
    if not obj.profiler_args:
        obj.profiler_args["activities"] = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
        obj.profiler_args["record_shapes"] = True

    if "on_trace_ready" not in obj.profiler_args:
        result_path = "/tmp/pytorch_profiler"
        dir_name = ""
        try:
            model_name = obj.manifest["model"]["modelName"]
            dir_name = model_name
        except KeyError:
            logging.debug("Model name not found in config")

        result_path = os.path.join(result_path, dir_name)
        obj.profiler_args["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(
            result_path
        )
        logger.info("Saving chrome trace to : %s", result_path)

    with profile(**obj.profiler_args) as prof:
        with record_function("preprocess"):
            data_preprocess = obj.preprocess(data)
        if not obj._is_explain():
            with record_function("inference"):
                output = obj.inference(data_preprocess)
            with record_function("postprocess"):
                output = obj.postprocess(output)
        else:
            with record_function("explain"):
                output = obj.explain_handle(data_preprocess, data)

    logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    return output, prof


class BasePreproc(PipeCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_preprocess


class BaseInit(InitCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_initialize


class BaseInference(PipeCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_inference


class BasePostprocess(PipeCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_postprocess


class BaseHandle(PipeCaller):
    def __init__(self, previous_handle=None):
        self._prev = previous_handle
        self._method = base_handle
