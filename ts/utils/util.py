"""
Utility functions for TorchServe
"""
import enum
import inspect
import itertools
import json
import logging
import os
import re


class PT2Backend(str, enum.Enum):
    EAGER = "eager"
    AOT_EAGER = "aot_eager"
    INDUCTOR = "inductor"
    NVFUSER = "nvfuser"
    AOT_NVFUSER = "aot_nvfuser"
    AOT_CUDAGRAPHS = "aot_cudagraphs"
    OFI = "ofi"
    FX2TRT = "fx2trt"
    ONNXRT = "onnxrt"
    IPEX = "ipex"


logger = logging.getLogger(__name__)

CLEANUP_REGEX = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")


def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.

    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [
        cls[1]
        for cls in inspect.getmembers(
            module,
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__,
        )
    ]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes


def load_compiler_config(config_file_path):
    """
    Load a compiler {compiler_name -> compiler }
    Can be extended to also support kwargs for ONNX and TensorRT
    """
    if not os.path.isfile(config_file_path):
        logger.info(f"{config_file_path} is missing. PT 2.0 will not be used")
        return None

    with open(config_file_path) as f:
        mapping = json.load(f)

    backend_values = [member.value for member in PT2Backend]
    if mapping["pt2"] in backend_values:
        return mapping["pt2"]
    else:
        logger.warning(f"{mapping['pt2']} is not a supported backend")
    return None


def load_label_mapping(mapping_file_path):
    """
    Load a JSON mapping { class ID -> friendly class name }.
    Used in BaseHandler.
    """
    if not os.path.isfile(mapping_file_path):
        logger.warning(
            f"{mapping_file_path!r} is missing. Inference output will not include class name."
        )
        return None

    with open(mapping_file_path) as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise Exception(
            'index->name JSON mapping should be in "class": "label" format.'
        )

    # Older examples had a different syntax than others. This code accommodates those.
    if "object_type_names" in mapping and isinstance(
        mapping["object_type_names"], list
    ):
        mapping = {str(k): v for k, v in enumerate(mapping["object_type_names"])}
        return mapping

    for key, value in mapping.items():
        new_value = value
        if isinstance(new_value, list):
            new_value = value[-1]
        if not isinstance(new_value, str):
            raise Exception(
                "labels in index->name mapping must be either str or List[str]"
            )
        mapping[key] = new_value
    return mapping


def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not isinstance(probs, list):
        raise Exception("Convert classes to list before doing mapping")

    if mapping is not None and not isinstance(mapping, dict):
        raise Exception("Mapping must be a dict")

    if lbl_classes is None:
        lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

    results = [
        {
            (mapping[str(lbl_class)] if mapping is not None else str(lbl_class)): prob
            for lbl_class, prob in zip(*row)
        }
        for row in zip(lbl_classes, probs)
    ]

    return results


class PredictionException(Exception):
    def __init__(self, message, error_code=500):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    def __str__(self):
        return f"{self.message} : {self.error_code}"
