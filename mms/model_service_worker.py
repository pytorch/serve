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
ModelServiceWorker is the worker that is started by the MMS front-end.
Communication message format: binary encoding
"""

# pylint: disable=redefined-builtin

import logging
import os
import platform
import socket
import sys

from mms.arg_parser import ArgParser
from mms.model_loader import ModelLoaderFactory
from mms.protocol.otf_message_handler import retrieve_msg, create_load_model_response
from mms.service import emit_metrics

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
DEBUG = False


class MXNetModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """
    def __init__(self, s_type=None, s_name=None, host_addr=None, port_num=None):
        if os.environ.get("OMP_NUM_THREADS") is None:
            os.environ["OMP_NUM_THREADS"] = "1"
        if os.environ.get("MXNET_USE_OPERATOR_TUNING") is None:
            # work around issue: https://github.com/apache/incubator-mxnet/issues/12255
            os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"

        self.sock_type = s_type
        if s_type == "unix":
            if s_name is None:
                raise ValueError("Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = s_name, -1
            try:
                os.remove(s_name)
            except OSError:
                if os.path.exists(s_name):
                    raise RuntimeError("socket already in use: {}.".format(s_name))

        elif s_type == "tcp":
            self.sock_name = host_addr if host_addr is not None else "127.0.0.1"
            if port_num is None:
                raise ValueError("Wrong arguments passed. No socket port given.")
            self.port = port_num
        else:
            raise ValueError("Incomplete data provided")

        logging.info("Listening on port: %s", s_name)
        socket_family = socket.AF_INET if s_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)

    @staticmethod
    def load_model(load_model_request):
        """
        Expected command
        {
            "command" : "load", string
            "modelPath" : "/path/to/model/file", string
            "modelName" : "name", string
            "gpu" : None if CPU else gpu_id, int
            "handler" : service handler entry point if provided, string
            "batchSize" : batch size, int
        }

        :param load_model_request:
        :return:
        """
        try:
            model_dir = load_model_request["modelPath"].decode("utf-8")
            model_name = load_model_request["modelName"].decode("utf-8")
            handler = load_model_request["handler"].decode("utf-8")
            batch_size = None
            if "batchSize" in load_model_request:
                batch_size = int(load_model_request["batchSize"])

            gpu = None
            if "gpu" in load_model_request:
                gpu = int(load_model_request["gpu"])

            model_loader = ModelLoaderFactory.get_model_loader(model_dir)
            service = model_loader.load(model_name, model_dir, handler, gpu, batch_size)

            logging.debug("Model %s loaded.", model_name)

            return service, "loaded model {}".format(model_name), 200
        except MemoryError:
            return None, "System out of memory", 507

    def handle_connection(self, cl_socket):
        """
        Handle socket connection.

        :param cl_socket:
        :return:
        """
        service = None
        while True:
            cmd, msg = retrieve_msg(cl_socket)
            if cmd == b'I':
                resp = service.predict(msg)
                cl_socket.send(resp)
            elif cmd == b'L':
                service, result, code = self.load_model(msg)
                resp = bytearray()
                resp += create_load_model_response(code, result)
                cl_socket.send(resp)
                if code != 200:
                    raise RuntimeError("{} - {}".format(code, result))
            else:
                raise ValueError("Received unknown command: {}".format(cmd))

            if service is not None and service.context is not None and service.context.metrics is not None:
                emit_metrics(service.context.metrics.store)

    def run_server(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """
        if not DEBUG:
            self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)

        if self.sock_type == "unix":
            self.sock.bind(self.sock_name)
        else:
            self.sock.bind((self.sock_name, int(self.port)))

        self.sock.listen(1)
        logging.info("[PID]%d", os.getpid())
        logging.info("MXNet worker started.")
        logging.info("Python runtime: %s", platform.python_version())

        while True:
            (cl_socket, _) = self.sock.accept()
            # workaround error(35, 'Resource temporarily unavailable') on OSX
            cl_socket.setblocking(True)

            logging.info("Connection accepted: %s.", cl_socket.getsockname())
            self.handle_connection(cl_socket)


if __name__ == "__main__":
    # Remove mms dir from python path to avoid module name conflict.
    mms_path = os.path.dirname(os.path.realpath(__file__))
    while mms_path in sys.path:
        sys.path.remove(mms_path)

    sock_type = None
    socket_name = None

    # noinspection PyBroadException
    try:
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
        args = ArgParser.model_service_worker_args().parse_args()
        socket_name = args.sock_name
        sock_type = args.sock_type
        host = args.host
        port = args.port

        worker = MXNetModelServiceWorker(sock_type, socket_name, host, port)
        worker.run_server()
    except socket.timeout:
        logging.error("Backend worker did not receive connection in: %d", SOCKET_ACCEPT_TIMEOUT)
    except Exception:  # pylint: disable=broad-except
        logging.error("Backend worker process die.", exc_info=True)
    finally:
        if sock_type == 'unix' and os.path.exists(socket_name):
            os.remove(socket_name)

    exit(1)
