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
Context object of incoming request
"""


class Context(object):
    """
    Context stores model relevant worker information
    Some fixed during load times and some
    """

    def __init__(self, model_name, model_dir, manifest, batch_size, gpu, mms_version):
        self.model_name = model_name
        self.manifest = manifest
        self._system_properties = {
            "model_dir": model_dir,
            "gpu_id": gpu,
            "batch_size": batch_size,
            "server_name": "MMS",
            "server_version": mms_version
        }
        self.request_ids = None
        self.request_processor = None
        self._metrics = None

    @property
    def system_properties(self):
        return self._system_properties

    @property
    def request_processor(self):
        return self._request_processor

    @request_processor.setter
    def request_processor(self, request_processor):
        self._request_processor = request_processor

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def get_request_id(self, idx=0):
        return self.request_ids.get(idx)

    def get_request_header(self, idx, key):
        return self._request_processor[idx].get_request_property(key)

    def get_all_request_header(self, idx):
        return self._request_processor[idx].get_request_properties()

    def set_response_content_type(self, idx, value):
        self.set_response_header(idx, 'content-type', value)

    def get_response_content_type(self, idx):
        return self.get_response_headers(idx).get('content-type')

    def get_response_status(self, idx):
        return self._request_processor[idx].get_response_status_code(), \
               self._request_processor[idx].get_response_status_phrase()

    def set_response_status(self, code=200, phrase="", idx=0):
        """
        Set the status code of individual requests
        :param phrase:
        :param idx: The index data in the list(data) that is sent to the handle() method
        :param code:
        :return:
        """
        if self._request_processor is not None and self._request_processor[idx] is not None:
            self._request_processor[idx].report_status(code,
                                                       reason_phrase=phrase)

    def set_all_response_status(self, code=200, phrase=""):
        """
        Set the status code of individual requests
        :param phrase:
        :param code:
        :return:
        """
        for idx, _ in enumerate(self._request_processor):
            self._request_processor[idx].report_status(code, reason_phrase=phrase)

    def get_response_headers(self, idx):
        return self._request_processor[idx].get_response_headers()

    def set_response_header(self, idx, key, value):
        self._request_processor[idx].add_response_property(key, value)

    # TODO: Should we add "add_header()" interface, to have multiple values for a single header. EG: Accept headers.

    def __eq__(self, other):
        return isinstance(other, Context) and self.__dict__ == other.__dict__


class RequestProcessor(object):
    """
    Request processor
    """

    def __init__(self, request_header):
        self._status_code = 200
        self._reason_phrase = None
        self._response_header = {}
        self._request_header = request_header

    def get_request_property(self, key):
        return self._request_header.get(key)

    def report_status(self, code, reason_phrase=None):
        self._status_code = code
        self._reason_phrase = reason_phrase

    def get_response_status_code(self):
        return self._status_code

    def get_response_status_phrase(self):
        return self._reason_phrase

    def add_response_property(self, key, value):
        self._response_header[key] = value

    def get_response_headers(self):
        return self._response_header

    def get_response_header(self, key):
        return self._response_header.get(key)

    def get_request_properties(self):
        return self._request_header
