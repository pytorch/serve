from functools import partial

import torch
import torch.nn.functional as F

from ts.handler_utils.caller import PipeCaller
from ts.utils.util import map_class_to_label


def postprocess(topk, self, data):
    ps = F.softmax(data, dim=1)
    probs, classes = torch.topk(ps, topk, dim=1)
    probs = probs.tolist()
    classes = classes.tolist()
    return map_class_to_label(probs, self.mapping, classes)


class ImageClassificationPostprocess(PipeCaller):
    def __init__(self, previous_handle=None, topk=5):
        self._prev = previous_handle
        self._method = partial(postprocess, topk)
