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
CustomService class definitions
"""
import logging
import time

from builtins import str

import mms
from mms.context import Context, RequestProcessor
from mms.metrics.metrics_store import MetricsStore
from mms.protocol.otf_message_handler import create_predict_response

PREDICTION_METRIC = 'PredictionTime'
logger = logging.getLogger(__name__)


class Service(object):
    """
    Wrapper for custom entry_point
    """

    def __init__(self, model_name, model_dir, manifest, entry_point, gpu, batch_size):
        self._context = Context(model_name, model_dir, manifest, batch_size, gpu, mms.__version__)
        self._entry_point = entry_point

    @property
    def context(self):
        return self._context

    @staticmethod
    def retrieve_data_for_inference(batch):
        """

        REQUEST_INPUT = {
            "requestId" : "111-222-3333",
            "parameters" : [ PARAMETER ]
        }

        PARAMETER = {
            "name" : parameter name
            "contentType": "http-content-types",
            "value": "val1"
        }

        :param batch:
        :return:
        """
        if batch is None:
            raise ValueError("Received invalid inputs")

        req_to_id_map = {}
        headers = []
        input_batch = []
        for batch_idx, request_batch in enumerate(batch):
            req_id = request_batch.get('requestId').decode("utf-8")
            parameters = request_batch['parameters']
            model_in_headers = dict()

            model_in = dict()
            # Parameter level headers are updated here. multipart/form-data can have multiple headers.
            for parameter in parameters:
                model_in.update({parameter["name"]: parameter["value"]})
                model_in_headers.update({parameter["name"]: {"content-type": parameter["contentType"]}})

            # Request level headers are populated here
            if request_batch.get("headers") is not None:
                for h in request_batch.get("headers"):
                    model_in_headers.update({h['name'].decode('utf-8'): h['value'].decode('utf-8')})

            headers.append(RequestProcessor(model_in_headers))
            input_batch.append(model_in)
            req_to_id_map[batch_idx] = req_id

        return headers, input_batch, req_to_id_map

    def predict(self, batch):
        """
        PREDICT COMMAND = {
            "command": "predict",
            "batch": [ REQUEST_INPUT ]
        }
        :param batch: list of request
        :return:

        """
        headers, input_batch, req_id_map = Service.retrieve_data_for_inference(batch)

        self.context.request_ids = req_id_map
        self.context.request_processor = headers
        metrics = MetricsStore(req_id_map, self.context.model_name)
        self.context.metrics = metrics

        start_time = time.time()

        # noinspection PyBroadException
        try:
            ret = self._entry_point(input_batch, self.context)
        except MemoryError:
            logger.error("System out of memory", exc_info=True)
            return create_predict_response(None, req_id_map, "Out of resources", 507)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Invoking custom service failed.", exc_info=True)
            return create_predict_response(None, req_id_map, "Prediction failed", 503)

        if not isinstance(ret, list):
            logger.warning("model: %s, Invalid return type: %s.", self.context.model_name, type(ret))
            return create_predict_response(None, req_id_map, "Invalid model predict output", 503)

        if len(ret) != len(input_batch):
            logger.warning("model: %s, number of batch response mismatched, expect: %d, got: %d.",
                           self.context.model_name, len(input_batch), len(ret))
            return create_predict_response(None, req_id_map, "number of batch response mismatched", 503)

        duration = round((time.time() - start_time) * 1000, 2)
        metrics.add_time(PREDICTION_METRIC, duration)

        return create_predict_response(ret, req_id_map, "Prediction success", 200, context=self.context)


def emit_metrics(metrics):
    """
    Emit the metrics in the provided Dictionary

    Parameters
    ----------
    metrics: Dictionary
    A dictionary of all metrics, when key is metric_name
    value is a metric object
    """
    if metrics:
        for met in metrics:
            logger.info("[METRICS]%s", str(met))
