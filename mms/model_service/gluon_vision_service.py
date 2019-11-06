# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""`Gluon vision service` defines a Gluon base vision service
"""
import numpy as np
import mxnet
from mms.model_service.mxnet_model_service import GluonImperativeBaseService
from mms.utils.mxnet import ndarray


class GluonVisionService(GluonImperativeBaseService):
    """MXNetVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """
    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = mxnet.img.imdecode(img)
            img_arr = mxnet.image.imresize(img_arr, w, h)
            img_arr = img_arr.astype(np.float32)
            img_arr /= 255
            img_arr = mxnet.image.color_normalize(img_arr,
                                                  mean=mxnet.nd.array([0.485, 0.456, 0.406]),
                                                  std=mxnet.nd.array([0.229, 0.224, 0.225]))
            img_arr = mxnet.nd.transpose(img_arr, (2, 0, 1))
            img_arr = img_arr.expand_dims(axis=0)
            img_list.append(img_arr)
        return img_list

    def _inference(self, data):
        """
        Internal inference methods for MMS service. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
               Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        """
        # Check input shape
        super(GluonVisionService, self)._inference(data)
        output = self.net(data[0])
        return output.softmax()

    def _postprocess(self, data):
        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [ndarray.top_probability(d, self.labels, top=5) for d in data]
