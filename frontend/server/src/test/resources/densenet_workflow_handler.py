import io
import logging
import os

from PIL import Image
import torch.nn.functional as F
import torch
import itertools
import json

from torchvision import transforms

logger = logging.getLogger(__name__)

topk = 5
# These are the standard Imagenet dimensions
# and statistics
image_processing = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pre_processing(data, context):
    if data:
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            image = image_processing(image)
            images.append(image)

        return images


def post_processing(data, context):
    if data:
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")
        data = torch.load(io.BytesIO(data))

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        mapping = load_label_mapping(mapping_file_path)

        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, mapping, classes)


def load_label_mapping(mapping_file_path):
    """
    Load a JSON mapping { class ID -> friendly class name }.
    Used in BaseHandler.
    """
    if not os.path.isfile(mapping_file_path):
        logger.warning(
            "Missing the index_to_name.json file. Inference output will not include class name."
        )
        return None

    with open(mapping_file_path) as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise Exception(
            'index_to_name mapping should be in "class":"label" json format'
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
            raise Exception("labels in index_to_name must be either str or [str]")
        mapping[key] = new_value
    return mapping


def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not (isinstance(probs, list) and isinstance(probs, list)):
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
