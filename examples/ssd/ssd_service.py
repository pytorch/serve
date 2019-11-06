# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np

from mxnet_utils import image
from mxnet_vision_service import MXNetVisionService


class SSDService(MXNetVisionService):
    """
    SSD Service to perform real time multi-object detection using pre-trained MXNet SSD model.
    This class extends MXNetVisionService to add custom preprocessing of input
    and preparing the output.
    Reuses input image transformation functionality of MXNetVisionService.
    """
    def __init__(self):
        super(SSDService, self).__init__()

        # Threshold is used to pick the detection boxes with score > threshold.
        # The detections from this network will be of the format - [[class_id, score, x1, y1, x2, y2]].
        # We pick all detections where 'score > threshold'.
        # You can experiment with different threshold to see the best threshold for the use-case.
        self.threshold = 0.2

        # This is used to save the original input image shape.
        # This is required for preparing the bounding box of the detected object "relative to
        # original input"
        self.input_width = None
        self.input_height = None

    def preprocess(self, batch):
        """
        Input image buffer from data is read into NDArray. Then, resized to
        expected shape. Swaps axes to convert image from BGR format to RGB.
        Returns the preprocessed NDArray as a list for next step, Inference.
        """

        # Read input
        img = batch[0].get("data")
        if img is None:
            img = batch[0].get("body")

        input_image = image.read(img)

        # Save original input image shape.
        # This is required for preparing the bounding box of the detected object relative to
        # original input
        self.input_height = input_image.shape[0]
        self.input_width = input_image.shape[1]

        # Transform input image - resize, BGR to RGB.
        # Reuse MXNetVisionService preprocess to achieve above transformations.
        return super(SSDService, self).preprocess(batch)

    def postprocess(self, data):
        """
        From the detections, prepares the output in the format of list of
        [(object_class, xmin, ymin, xmax, ymax)]
        object_class is name of the object detected. xmin, ymin, xmax, ymax
        provides the bounding box coordinates.

        Example: [(person, 555, 175, 581, 242), (dog, 306, 446, 468, 530)]
        """

        # Read the detections output after forward pass (inference)
        detections = data[0].asnumpy()
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)

        # Prepare the output
        dets = result[0]
        classes = self.labels
        width = self.input_width    # original input image width
        height = self.input_height  # original input image height
        response = []
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > self.threshold:
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    response.append((class_name, xmin, ymin, xmax, ymax))
        return [response]


_service = SSDService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
