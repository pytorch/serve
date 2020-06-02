

"""
ModelServiceWorker is the worker that is started by the MMS front-end.
Communication message format: binary encoding
"""

# pylint: disable=redefined-builtin

import logging
import os
import multiprocessing
import platform
import socket
import sys
import signal

from ts.arg_parser import ArgParser
from ts.model_loader import ModelLoaderFactory
from ts.protocol.otf_message_handler import retrieve_msg, create_load_model_response
from ts.service import emit_metrics

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
DEBUG = False
BENCHMARK = False


class TorchModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """

    def __init__(self, s_type=None, s_name=None, host_addr=None, port_num=None,
                 model_request=None, preload_model=False, tmp_dir="/tmp"):
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
            raise ValueError("Invalid socket type provided")

        logging.info("Listening on port: %s", s_name)
        socket_family = socket.AF_INET if s_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)

        self.preload = preload_model
        self.service = None
        self.model_meta_data = model_request
        self.out = self.err = None
        self.tmp_dir = tmp_dir
        self.socket_name = s_name

    def load_model(self, load_model_request=None):
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
            handler = load_model_request["handler"].decode("utf-8") if load_model_request["handler"] else None
            batch_size = 1
            if "batchSize" in load_model_request:
                batch_size = int(load_model_request["batchSize"])

            gpu = None
            if "gpu" in load_model_request:
                gpu = int(load_model_request["gpu"])

            io_fd = None
            if "ioFileDescriptor" in load_model_request:
                io_fd = load_model_request.get("ioFileDescriptor").decode("utf-8")
                self._create_io_files(self.tmp_dir, io_fd)

            if self.service is None or self.preload is False:
                model_loader = ModelLoaderFactory.get_model_loader()
                self.service = model_loader.load(model_name, model_dir, handler, gpu, batch_size)
                logging.info("Model %s loaded io_fd=%s", model_name, str(io_fd))

            return "loaded model {}. [PID]:{}".format(model_name, os.getpid()), 200
        except MemoryError:
            return None, "System out of memory", 507

    def _create_io_files(self, tmp_dir, io_fd):
        self.out = tmp_dir + '/' + io_fd + "-stdout"

        self.err = tmp_dir + '/' + io_fd + "-stderr"
        # TODO: Windows support
        os.mkfifo(self.out)
        os.mkfifo(self.err)

    def _remap_io(self):
        out_fd = open(self.out, "w")

        err_fd = open(self.err, "w")
        os.dup2(out_fd.fileno(), sys.stdout.fileno())
        os.dup2(err_fd.fileno(), sys.stderr.fileno())

    def handle_connection(self, cl_socket):
        """
        Handle socket connection.

        :param cl_socket:
        :return:
        """
        cl_socket.setblocking(True)
        while True:
            if BENCHMARK:
                pr.disable()
                pr.dump_stats('/tmp/tsPythonProfile.prof')
            cmd, msg = retrieve_msg(cl_socket)
            if BENCHMARK:
                pr.enable()
            if cmd == b'I':
                resp = self.service.predict(msg)
                cl_socket.send(resp)
            elif cmd == b'L':
                result, code = self.load_model(msg)
                resp = bytearray()
                resp += create_load_model_response(code, result)
                cl_socket.send(resp)
                self._remap_io()
                if code != 200:
                    raise RuntimeError("{} - {}".format(code, result))
            else:
                raise ValueError("Received unknown command: {}".format(cmd))

            if self.service is not None and self.service.context is not None \
                    and self.service.context.metrics is not None:
                emit_metrics(self.service.context.metrics.store)

    def sigterm_handler(self):
        for node in [self.socket_name, self.out, self.err]:
            try:
                os.remove(node)
            except OSError:
                pass

    def start_worker(self, cl_socket):
        """
        Method to start the worker threads. These worker threads use multiprocessing to spawn a new worker.
        :param cl_socket:
        :return:
        """

        self.sock.close()  # close listening socket in the fork

        try:
            signal.signal(signal.SIGTERM, lambda signum, frame: self.sigterm_handler())
            self.handle_connection(cl_socket)
        except Exception:  # pylint: disable=broad-except
            logging.error("Backend worker process die.", exc_info=True)
        finally:
            try:
                os.remove(self.out)
                os.remove(self.err)
            finally:
                cl_socket.shutdown(socket.SHUT_RDWR)
                cl_socket.close()
                os._exit(1)

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

        self.sock.listen(128)
        logging.info("[PID] %d", os.getpid())
        logging.info("Torch worker started.")
        logging.info("Python runtime: %s", platform.python_version())

        while True:
            if self.service is None and self.preload is True:
                # Lazy loading the models
                self.load_model(self.model_meta_data)
            (cl_socket, _) = self.sock.accept()
            # workaround error(35, 'Resource temporarily unavailable') on OSX
            cl_socket.setblocking(True)

            logging.info("Connection accepted: %s.", cl_socket.getsockname())
            p = multiprocessing.Process(target=self.start_worker, args=(cl_socket,))
            p.start()
            cl_socket.close()


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
        logging.info("model_service_worker started with args: %s", " ".join(sys.argv[1:]))
        model_req = dict()
        args = ArgParser.model_service_worker_args().parse_args()
        socket_name = args.sock_name
        sock_type = args.sock_type
        host = args.host
        port = args.port

        if BENCHMARK:
            import cProfile
            pr = cProfile.Profile()
            pr.disable()
            pr.dump_stats('/tmp/tsPythonProfile.prof')

        model_req["handler"] = args.handler.encode('utf-8')
        model_req["modelPath"] = args.model_path.encode('utf-8')
        model_req["modelName"] = args.model_name.encode('utf-8')
        worker = TorchModelServiceWorker(sock_type, socket_name, host, port, model_req,
                                         args.preload_model, args.tmp_dir)
        worker.run_server()
        if BENCHMARK:
            pr.disable()
            pr.dump_stats('/tmp/tsPythonProfile.prof')

    except socket.timeout:
        logging.error("Backend worker did not receive connection in: %d", SOCKET_ACCEPT_TIMEOUT)
    except Exception:  # pylint: disable=broad-except
        logging.error("Backend worker process die.", exc_info=True)
    finally:
        if sock_type == 'unix' and os.path.exists(socket_name):
            os.remove(socket_name)

    sys.exit(1)
