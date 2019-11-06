# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Gluon Base service defines a Gluon base service for generic CNN
"""
import mxnet as mx
import numpy as np
import os
import json
import ndarray


class GluonBaseService(object):
    """GluonBaseService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """

    def __init__(self):
        self.param_filename = None
        self.model_name = None
        self.initialized = False
        self.ctx = None
        self.net = None
        self._signature = None
        self.labels = None
        self.signature = None

    def initialize(self, params):
        """
        Initialization of the network
        :param params: This is the :func `Context` object
        :return:
        """
        if self.net is None:
            raise NotImplementedError("Gluon network not defined")
        sys_prop = params.system_properties
        gpu_id = sys_prop.get("gpu_id")
        model_dir = sys_prop.get("model_dir")
        self.model_name = params.manifest["model"]["modelName"]
        self.ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        if self.param_filename is not None:
            param_file_path = os.path.join(model_dir, self.param_filename)
            if not os.path.isfile(param_file_path):
                raise OSError("Parameter file not found {}".format(param_file_path))
            self.net.load_parameters(param_file_path, self.ctx)

        synset_file = os.path.join(model_dir, "synset.txt")
        signature_file_path = os.path.join(model_dir, "signature.json")

        if not os.path.isfile(signature_file_path):
            raise OSError("Signature file not found {}".format(signature_file_path))

        if not os.path.isfile(synset_file):
            raise OSError("synset file not available {}".format(synset_file))

        with open(signature_file_path) as sig_file:
            self.signature = json.load(sig_file)
            
        self.labels = [line.strip() for line in open(synset_file).readlines()]
        self.initialized = True

    def preprocess(self, data):
        """
        This method considers only one input data

        :param data: Data is list of map
        format is
        [
        {
            "parameterName": name
            "parameterValue": data
        },
        {...}
        ]
        :return:
        """

        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']

        img = data[0].get(param_name)

        if img is None:
            raise IOError("Invalid parameter given")

        # We are assuming input shape is NCHW
        [h, w] = input_shape[2:]
        img_arr = mx.img.imdecode(img)
        img_arr = mx.image.imresize(img_arr, w, h)
        img_arr = img_arr.astype(np.float32)
        img_arr /= 255
        img_arr = mx.image.color_normalize(img_arr,
                                           mean=mx.nd.array([0.485, 0.456, 0.406]),
                                           std=mx.nd.array([0.229, 0.224, 0.225]))
        img_arr = mx.nd.transpose(img_arr, (2, 0, 1))
        img_arr = img_arr.expand_dims(axis=0)
        return img_arr

    def inference(self, data):
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
        model_input = data.as_in_context(self.ctx)
        output = self.net(model_input)
        return output.softmax()

    def postprocess(self, data):
        assert hasattr(self, 'labels'), \
            "Can't find labels attribute. Did you put synset.txt file into " \
            "model archive or manually load class label file in __init__?"
        return [[ndarray.top_probability(d, self.labels, top=5) for d in data]]

    def predict(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        return self.postprocess(data)
