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
from typing import Optional

from ts.arg_parser import ArgParser
from ts.metrics.metric_cache_yaml_impl import MetricsCacheYamlImpl
from ts.model_loader import ModelLoaderFactory
from ts.protocol.otf_message_handler import create_load_model_response, retrieve_msg

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
DEBUG = False
BENCHMARK = os.getenv("TS_BENCHMARK") in ["True", "true", "TRUE"]
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 0))
WORLD_RANK = int(os.getenv("RANK", 0))
LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", 0))


class TorchModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """

    def __init__(
        self,
        s_type: Optional[str] = None,
        s_name: Optional[str] = None,
        host_addr: Optional[str] = None,
        port_num: Optional[int] = None,
        metrics_config: Optional[str] = None,
    ):
        self.sock_type = s_type

        if s_type == "unix":
            if s_name is None:
                raise ValueError("Wrong arguments passed. No socket name given.")
            s_name_parts = s_name.rsplit(".", 1)
            logging.info(
                "s_name_part0=%s, s_name_part1=%s, pid=%d",
                s_name_parts[0],
                s_name_parts[1],
                os.getpid(),
            )
            s_name_new = s_name_parts[0] + "." + str(int(s_name_parts[1]) + LOCAL_RANK)
            self.sock_name, self.port = s_name_new, -1
            try:
                os.remove(s_name_new)
            except OSError as e:
                if os.path.exists(s_name_new):
                    raise RuntimeError(
                        "socket already in use: {}.".format(s_name_new)
                    ) from e
            logging.info("Listening on port: %s", s_name_new)
        elif s_type == "tcp":
            self.sock_name = host_addr if host_addr is not None else "127.0.0.1"
            if port_num is None:
                raise ValueError("Wrong arguments passed. No socket port given.")
            self.port = int(port_num) + LOCAL_RANK
            logging.info("Listening on addr:port: %s:%d", self.sock_name, self.port)
        else:
            raise ValueError("Incomplete data provided")

        socket_family = socket.AF_INET if s_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)
        self.metrics_cache = MetricsCacheYamlImpl(config_file_path=metrics_config)
        if self.metrics_cache:
            self.metrics_cache.initialize_cache()
        else:
            raise RuntimeError(
                f"Failed to initialize metrics from file {metrics_config}"
            )

    def load_model(self, load_model_request):
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

            self.metrics_cache.model_name = model_name
            model_loader = ModelLoaderFactory.get_model_loader()
            service = model_loader.load(
                model_name,
                model_dir,
                handler,
                gpu,
                batch_size,
                envelope,
                limit_max_image_pixels,
                self.metrics_cache,
            )

            logging.debug("Model %s loaded.", model_name)

            return service, "loaded model {}".format(model_name), 200
        except MemoryError as ex:
            logging.exception(
                "Load model %s cpu OOM, exception %s", model_name, str(ex)
            )
            return None, "System out of memory", 507
        except RuntimeError as ex:  # pylint: disable=broad-except
            if "CUDA" in str(ex):
                # Handles Case A: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED (Close to OOM) &
                # Case B: CUDA out of memory (OOM)
                logging.exception(
                    "Load model %s cuda OOM, exception %s", model_name, str(ex)
                )
                return None, "System out of memory", 507
            else:
                # Sanity testcases fail without this
                logging.exception(
                    "Failed to load model %s, exception %s", model_name, str(ex)
                )
                return None, "Unknown exception", 500

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
                if service is not None:
                    resp = service.predict(msg)
                    cl_socket.sendall(resp)
                else:
                    raise RuntimeError(
                        "Received command: {}, but service is not loaded".format(cmd)
                    )
            elif cmd == b"L":
                service, result, code = self.load_model(msg)
                resp = bytearray()
                resp += create_load_model_response(code, result)
                cl_socket.sendall(resp)
                if code != 200:
                    raise RuntimeError("{} - {}".format(code, result))
                service.set_cl_socket(cl_socket)
            else:
                raise ValueError("Received unknown command: {}".format(cmd))

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

    sock_type: Optional[str] = None
    socket_name: Optional[str] = None

    # noinspection PyBroadException
    try:
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
        args = ArgParser.model_service_worker_args().parse_args()
        socket_name = args.sock_name
        sock_type = args.sock_type
        host = args.host
        port = args.port
        metrics_config = args.metrics_config

        if BENCHMARK:
            import cProfile

            pr = cProfile.Profile()
            pr.disable()
            pr.dump_stats("/tmp/tsPythonProfile.prof")

        worker = TorchModelServiceWorker(
            sock_type, socket_name, host, port, metrics_config
        )
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
        if (
            sock_type == "unix"
            and socket_name is not None
            and os.path.exists(socket_name)
        ):
            os.remove(socket_name)

    sys.exit(1)
