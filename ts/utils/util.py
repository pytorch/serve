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
from functools import wraps
from warnings import warn

import yaml


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
    TORCHXLA_TRACE_ONCE = "torchxla_trace_once"


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


def check_valid_pt2_backend(backend: str) -> bool:
    backend_values = [member.value for member in PT2Backend]
    if backend in backend_values:
        return True
    else:
        logger.warning(f"{backend} is not a supported backend")
    return False


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


def get_yaml_config(yaml_file_path):
    config = {}
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


class PredictionException(Exception):
    def __init__(self, message, error_code=500):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    def __str__(self):
        return f"{self.message} : {self.error_code}"


def deprecated(version, replacement="", klass=PendingDeprecationWarning):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Args:
        version: The version in which the function will be removed.
        replacement: The replacement function, if any.
        klass: The category of warning
    """

    def deprecator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warn(
                f"{func.__name__} is deprecated in {version} and moved to {replacement}",
                klass,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return deprecator
