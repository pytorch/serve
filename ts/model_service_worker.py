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

from ts.arg_parser import ArgParser
from ts.model_loader import ModelLoaderFactory
from ts.protocol.otf_message_handler import retrieve_msg, create_load_model_response
from ts.service import emit_metrics

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
DEBUG = False
BENCHMARK = os.getenv("TS_BENCHMARK")
BENCHMARK = BENCHMARK in ["True", "true", "TRUE"]


class TorchModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """

    def __init__(self, s_type=None, s_name=None, host_addr=None, port_num=None):
        self.sock_type = s_type
        if s_type == "unix":
            if s_name is None:
                raise ValueError("Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = s_name, -1
            try:
                os.remove(s_name)
            except OSError as e:
                if os.path.exists(s_name):
                    raise RuntimeError(
                        "socket already in use: {}.".format(s_name)
                    ) from e

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
            "envelope" : name of wrapper/unwrapper of request data if provided, string
            "batchSize" : batch size, int
            "limitMaxImagePixels": limit pillow image max_image_pixels, bool
        }

        :param load_model_request:
        :return:
        """
        try:
            model_dir = load_model_request["modelPath"].decode("utf-8")
            model_name = load_model_request["modelName"].decode("utf-8")
            handler = (
                load_model_request["handler"].decode("utf-8")
                if load_model_request["handler"]
                else None
            )
            envelope = (
                load_model_request["envelope"].decode("utf-8")
                if "envelope" in load_model_request
                else None
            )
            envelope = envelope if envelope is not None and len(envelope) > 0 else None

            batch_size = None
            if "batchSize" in load_model_request:
                batch_size = int(load_model_request["batchSize"])
            logging.info("model_name: %s, batchSize: %d", model_name, batch_size)

            gpu = None
            if "gpu" in load_model_request:
                gpu = int(load_model_request["gpu"])

            limit_max_image_pixels = True
            if "limitMaxImagePixels" in load_model_request:
                limit_max_image_pixels = bool(load_model_request["limitMaxImagePixels"])

            model_loader = ModelLoaderFactory.get_model_loader()
            service = model_loader.load(
                model_name,
                model_dir,
                handler,
                gpu,
                batch_size,
                envelope,
                limit_max_image_pixels,
            )

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
            if BENCHMARK:
                pr.disable()
                pr.dump_stats("/tmp/tsPythonProfile.prof")
            cmd, msg = retrieve_msg(cl_socket)
            if BENCHMARK:
                pr.enable()
            if cmd == b"I":
                resp = service.predict(msg)
                cl_socket.sendall(resp)
            elif cmd == b"L":
                service, result, code = self.load_model(msg)
                resp = bytearray()
                resp += create_load_model_response(code, result)
                cl_socket.sendall(resp)
                if code != 200:
                    raise RuntimeError("{} - {}".format(code, result))
            else:
                raise ValueError("Received unknown command: {}".format(cmd))

            if (
                service is not None
                and service.context is not None
                and service.context.metrics is not None
            ):
                emit_metrics(service.context.metrics.store)

    def run_server(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """
        if not DEBUG:
            self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)

        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self.sock_type == "unix":
            self.sock.bind(self.sock_name)
        else:
            self.sock.bind((self.sock_name, int(self.port)))

        self.sock.listen(1)
        logging.info("[PID]%d", os.getpid())
        logging.info("Torch worker started.")
        logging.info("Python runtime: %s", platform.python_version())

        while True:
            (cl_socket, _) = self.sock.accept()
            # workaround error(35, 'Resource temporarily unavailable') on OSX
            cl_socket.setblocking(True)

            logging.info("Connection accepted: %s.", cl_socket.getsockname())
            self.handle_connection(cl_socket)


if __name__ == "__main__":
    # Remove ts dir from python path to avoid module name conflict.
    ts_path = os.path.dirname(os.path.realpath(__file__))
    while ts_path in sys.path:
        sys.path.remove(ts_path)

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

        if BENCHMARK:
            import cProfile

            pr = cProfile.Profile()
            pr.disable()
            pr.dump_stats("/tmp/tsPythonProfile.prof")

        worker = TorchModelServiceWorker(sock_type, socket_name, host, port)
        worker.run_server()
        if BENCHMARK:
            pr.disable()
            pr.dump_stats("/tmp/tsPythonProfile.prof")

    except socket.timeout:
        logging.error(
            "Backend worker did not receive connection in: %d", SOCKET_ACCEPT_TIMEOUT
        )
    except Exception:  # pylint: disable=broad-except
        logging.error("Backend worker process died.", exc_info=True)
    finally:
        if sock_type == "unix" and os.path.exists(socket_name):
            os.remove(socket_name)

    sys.exit(1)
