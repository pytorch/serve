# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet as mx
import json
import os
import numpy as np
from collections import namedtuple
import logging


class MXNetVisionServiceBatching(object):
    def __init__(self):
        """
        Initialization for MXNet Vision Service supporting batch inference
        """
        self.mxnet_ctx = None
        self.mx_model = None
        self.labels = None
        self.signature = None
        self.epoch = 0
        self._context = None
        self._batch_size = 0
        self.initialized = False
        self.erroneous_reqs = set()

    def top_probability(self, data, labels, top=5):
        """
        Get top probability prediction from NDArray.

        :param data: NDArray
            Data to be predicted
        :param labels: List
            List of class labels
        :param top:
        :return: List
            List of probability: class pairs in sorted order
        """
        dim = len(data.shape)
        if dim > 2:
            data = mx.nd.array(
                np.squeeze(data.asnumpy(), axis=tuple(range(dim)[2:])))
        sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
        top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:top])
        return [{'probability': float(data[0, i].asscalar()), 'class': labels[i]}
                for i in top_prob]

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")

        with open(signature_file_path) as f:
            self.signature = json.load(f)

        model_files_prefix = context.manifest["model"]["modelName"]
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

        self.mx_model = mx.mod.Module(symbol=sym, context=self.mxnet_ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    def inference(self, model_input):
        """
        Internal inference methods for MXNet. Run forward computation and
        return output.

        :param model_input: list of NDArray
            Preprocessed inputs in NDArray format.
        :return: list of NDArray
            Inference output.
        """
        batch = namedtuple('Batch', ['data'])

        self.mx_model.forward(batch([model_input]), is_train=False)
        outputs = self.mx_model.get_outputs()
        res = mx.ndarray.split(outputs[0], axis=0, num_outputs=outputs[0].shape[0])
        res = [res] if not isinstance(res, list) else res
        return res

    def preprocess(self, request):
        """
        Decode all input images into ndarray.

        Note: This implementation doesn't properly handle error cases in batch mode,
        If one of the input images is corrupted, all requests in the batch will fail.

        :param request:
        :return:
        """
        img_list = []
        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']
        # We are assuming input shape is NCHW
        [c, h, w] = input_shape[1:]

        # Clear error requests set.
        self.erroneous_reqs.clear()

        for idx, data in enumerate(request):
            img = data.get(param_name)
            if img is None:
                img = data.get("body")

            if img is None:
                img = data.get("data")

            if img is None or len(img) == 0:
                logging.error("Error processing request")
                self.erroneous_reqs.add(idx)
                continue

            try:
                img_arr = mx.image.imdecode(img, 1, True, None)
            except Exception as e:
                logging.error(e, exc_info=True)
                self.erroneous_reqs.add(idx)
                continue

            img_arr = mx.image.imresize(img_arr, w, h, 2)
            img_arr = mx.nd.transpose(img_arr, (2, 0, 1))
            self._num_requests = idx + 1
            img_list.append(img_arr)
        
        logging.debug("Worker :{} received {} requests".format(os.getpid(), self._num_requests))
        reqs = mx.nd.stack(*img_list)
        reqs = reqs.as_in_context(self.mxnet_ctx)

        if (self._batch_size - self._num_requests) != 0:
            padding = mx.nd.zeros((self._batch_size - self._num_requests, c, h, w), self.mxnet_ctx, 'uint8')
            reqs = mx.nd.concat(reqs, padding, dim=0)

        return reqs

    def postprocess(self, data):
        res = []
        for idx, resp in enumerate(data[:self._num_requests]):
            if idx not in self.erroneous_reqs:
                res.append(self.top_probability(resp, self.labels, top=5))
            else:
                res.append("This request was not processed successfully. Refer to mms.log for additional information")
        return res


_service = MXNetVisionServiceBatching()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    try:
        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        logging.error(e, exc_info=True)
        request_processor = context.request_processor
        request_processor.report_status(500, "Unknown inference error")
        return [str(e)] * _service._batch_size
