

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
from datetime import datetime
import traceback


from ts.arg_parser import ArgParser
from ts.model_loader import ModelLoaderFactory
from ts.protocol.otf_message_handler import retrieve_msg, create_load_model_response
from ts.service import emit_metrics

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
DEBUG = False
BENCHMARK = os.getenv('TS_BENCHMARK')
BENCHMARK = BENCHMARK in ['True', 'true', 'TRUE']


class TorchModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """
    def __init__(self, s_type=None, s_name=None, host_addr=None, port_num=None, service=None, model_loader_args=None, fifo_path=None):


        self.sock_type = s_type
        if s_type == "unix":
            if s_name is None:
                raise ValueError("Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = s_name, -1
            try:
                os.remove(s_name)
            except OSError as e :
                if os.path.exists(s_name):
                    raise RuntimeError("socket already in use: {}.".format(s_name)) from e

        elif s_type == "tcp":
            self.sock_name = host_addr if host_addr is not None else "127.0.0.1"
            if port_num is None:
                raise ValueError("Wrong arguments passed. No socket port given.")
            self.port = port_num
        else:
            raise ValueError("Incomplete data provided")

        logging.info("Will listen on port: %s", s_name)
        socket_family = socket.AF_INET if s_type == "tcp" else socket.AF_UNIX
        self.sock = socket.socket(socket_family, socket.SOCK_STREAM)
        self.port_num = port_num
        self.service = service
        self.model_loader_args = model_loader_args
        self.fifo_path = fifo_path


    def handle_connection(self, cl_socket):
        """
        Handle socket connection.

        :param cl_socket:
        :return:
        """

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
            else:
                raise ValueError("Received unknown command: {}".format(cmd))

            if self.service is not None and self.service.context is not None and self.service.context.metrics is not None:
                emit_metrics(self.service.context.metrics.store)



    def run_server(self):
        try:
            logging.error("Run server invoke")
            loggerc = logging.getLogger('c')
            loggerc.addHandler(logging.FileHandler('/tmp/loggerc'))
            loggerc.error("Starting server ....  " + str(self.sock_name))
            self.run_server1()
        except:
            e = sys.exc_info()[0]
            traceback.print_exc()
            loggerc.error("after the listen....." + str(e) + str(self.sock_name))

    def run_server1(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """

        logger1 = logging.getLogger('1')
        logger1.addHandler(logging.FileHandler('/tmp/logger1'))
        logger1.addHandler(logging.FileHandler(self.fifo_path+".out"))
        logger1.error("testing logger ....." + self.fifo_path+".out")

        logging.basicConfig(format="%(message)s", filename=self.fifo_path+".out", filemode="a+", level=logging.INFO)

        self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)

        if self.sock_type == "unix":
            self.sock.bind(self.sock_name)
        else:
            self.sock.bind((self.sock_name, int(self.port)))

        logging.info("[PID]%d", os.getpid())
        logging.info("Torch worker started.")
        logging.info("Python runtime: %s", platform.python_version())


        logger1.error("[PID]%d", os.getpid())
        logger1.error("Torch worker started.")
        logger1.error("Python runtime: %s", platform.python_version())


        if(self.service is None):
            model_loader = ModelLoaderFactory.get_model_loader()
            self.service = model_loader.load(*self.model_loader_args)

        logger1.error("Waiting for connection with socket timeout... " + str(self.sock.gettimeout()))
        while True:
            try:
                logger1.error("listenning to socket ....." + str(datetime.now()))
                self.sock.listen(1)
                (cl_socket, _) = self.sock.accept()
                # workaround error(35, 'Resource temporarily unavailable') on OSX
                logger1.error("after the listen.....")
                cl_socket.setblocking(True)

                logging.info("Connection accepted: %s.", cl_socket.getsockname())
                self.handle_connection(cl_socket)
            except socket.timeout:
                logger1.error("socket time out ....." + str(datetime.now()))
                logging.info("Connection timedout")
                pass
