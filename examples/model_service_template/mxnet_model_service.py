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
MXNetModelService defines an API for MXNet service.
"""
import json
import os

import mxnet as mx
from mxnet.io import DataBatch

from model_handler import ModelHandler


class MXNetModelService(ModelHandler):
    """
    MXNetBaseService defines the fundamental loading model and inference
    operations when serving MXNet model. This is a base class and needs to be
    inherited.
    """

    def __init__(self):
        super(MXNetModelService, self).__init__()
        self.mxnet_ctx = None
        self.mx_model = None
        self.labels = None
        self.signature = None
        self.epoch = 0

    # noinspection PyMethodMayBeStatic
    def get_model_files_prefix(self, context):
        return context.manifest["model"]["modelName"]

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        super(MXNetModelService, self).initialize(context)

        assert self._batch_size == 1, "Batch is not supported."

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")

        with open(signature_file_path) as f:
            self.signature = json.load(f)

        model_files_prefix = self.get_model_files_prefix(context)
        archive_synset = os.path.join(model_dir, "synset.txt")
        if os.path.isfile(archive_synset):
            synset = archive_synset
            self.labels = [line.strip() for line in open(synset).readlines()]

        data_names = []
        data_shapes = []
        for input_data in self.signature["inputs"]:
            data_name = input_data["data_name"]
            data_shape = input_data["data_shape"]

            # Set batch size
            data_shape[0] = self._batch_size

            # Replace 0 entry in data shape with 1 for binding executor.
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1

            data_names.append(data_name)
            data_shapes.append((data_name, tuple(data_shape)))

        checkpoint_prefix = "{}/{}".format(model_dir, model_files_prefix)

        # Load MXNet module
        self.mxnet_ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)
        sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_prefix, self.epoch)

        # noinspection PyTypeChecker
        self.mx_model = mx.mod.Module(symbol=sym, context=self.mxnet_ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    def preprocess(self, batch):
        """
        Transform raw input into model input data.

        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))

        ret = []
        param_name = self.signature['inputs'][0]['data_name']

        for idx, request in enumerate(batch):
            data = request.get(param_name)
            if data is None:
                data = request.get("body")

            if data is None:
                data = request.get("data")

            ret.append(map(mx.nd.array, data))

        return ret

    def inference(self, model_input):
        """
        Internal inference methods for MXNet. Run forward computation and
        return output.

        :param model_input: list of NDArray
            Preprocessed inputs in NDArray format.
        :return: list of NDArray
            Inference output.
        """
        if self.error is not None:
            return None

        # Check input shape
        check_input_shape(model_input, self.signature)
        model_input = [item.as_in_context(self.mxnet_ctx) for item in model_input]
        self.mx_model.forward(DataBatch(model_input))
        model_input = self.mx_model.get_outputs()
        # by pass lazy evaluation get_outputs either returns a list of nd arrays
        # a list of list of NDArray
        for d in model_input:
            if isinstance(d, list):
                for n in model_input:
                    if isinstance(n, mx.ndarray.ndarray.NDArray):
                        n.wait_to_read()
            elif isinstance(d, mx.ndarray.ndarray.NDArray):
                d.wait_to_read()
        return model_input

    def postprocess(self, inference_output):
        if self.error is not None:
            return [self.error] * self._batch_size

        return [str(d.asnumpy().tolist()) for d in inference_output]


def check_input_shape(inputs, signature):
    """
    Check input data shape consistency with signature.

    Parameters
    ----------
    inputs : List of NDArray
        Input data in NDArray format.
    signature : dict
        Dictionary containing model signature.
    """
    assert isinstance(inputs, list), 'Input data must be a list.'
    assert len(inputs) == len(signature['inputs']), \
        "Input number mismatches with " \
        "signature. %d expected but got %d." \
        % (len(signature['inputs']), len(inputs))
    for input_data, sig_input in zip(inputs, signature["inputs"]):
        assert isinstance(input_data, mx.nd.NDArray), 'Each input must be NDArray.'
        assert len(input_data.shape) == len(sig_input["data_shape"]), \
            'Shape dimension of input %s mismatches with ' \
            'signature. %d expected but got %d.' \
            % (sig_input['data_name'],
               len(sig_input['data_shape']),
               len(input_data.shape))
        for idx in range(len(input_data.shape)):
            if idx != 0 and sig_input['data_shape'][idx] != 0:
                assert sig_input['data_shape'][idx] == input_data.shape[idx], \
                    'Input %s has different shape with ' \
                    'signature. %s expected but got %s.' \
                    % (sig_input['data_name'], sig_input['data_shape'],
                       input_data.shape)
