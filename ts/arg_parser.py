"""
This module parses the arguments given through the torchserve command-line. This is used by model-server
at runtime.
"""

import argparse


# noinspection PyTypeChecker
class ArgParser(object):
    """
    Argument parser for torchserve and torchserve-export commands
    TODO : Add readme url
    """

    @staticmethod
    def ts_parser():
        """
        Argument parser for torchserve start service
        """
        parser = argparse.ArgumentParser(prog="torchserve", description="Torchserve")

        sub_parse = parser.add_mutually_exclusive_group(required=False)
        sub_parse.add_argument(
            "-v", "--version", action="store_true", help="Return TorchServe Version"
        )
        sub_parse.add_argument(
            "--start", action="store_true", help="Start the model-server"
        )
        sub_parse.add_argument(
            "--stop", action="store_true", help="Stop the model-server"
        )

        parser.add_argument(
            "--ts-config", dest="ts_config", help="Configuration file for model server"
        )
        parser.add_argument(
            "--model-store",
            required=False,
            dest="model_store",
            help="Model store location from where local or default models can be loaded",
        )
        parser.add_argument(
            "--workflow-store",
            required=False,
            dest="workflow_store",
            help="Workflow store location from where local or default workflows can be loaded",
        )
        parser.add_argument(
            "--models",
            metavar="MODEL_PATH1 MODEL_NAME=MODEL_PATH2...",
            nargs="+",
            help="Models to be loaded using [model_name=]model_location format. "
            "Location can be a HTTP URL or a model archive file  in MODEL_STORE.",
        )
        parser.add_argument(
            "--log-config",
            dest="log_config",
            help="Log4j configuration file for model server",
        )
        parser.add_argument(
            "--cpp-log-config",
            dest="cpp_log_config",
            help="log configuration file for cpp backend",
        )
        parser.add_argument(
            "--foreground",
            help="Run the model server in foreground. If this option is disabled, the model server"
            " will run in the background.",
            action="store_true",
        )
        parser.add_argument(
            "--no-config-snapshots",
            "--ncs",
            dest="no_config_snapshots",
            help="Prevents to server from storing config snapshot files.",
            action="store_true",
        )
        parser.add_argument(
            "--plugins-path",
            "--ppath",
            dest="plugins_path",
            help="plugin jars to be included in torchserve class path",
        )
        parser.add_argument(
            "--disable-token-auth",
            "--dt",
            dest="token_auth",
            help="if this option is set then token authorization is disabled",
            action="store_true",
        )
        parser.add_argument(
            "--enable-model-api",
            dest="model_mode",
            help="enables model control apis",
            action="store_true",
        )

        return parser

    @staticmethod
    def model_service_worker_args():
        """
        ArgParser for backend worker. Takes the socket name and socket type.
        :return:
        """
        parser = argparse.ArgumentParser(
            prog="model-server-worker", description="Model Server Worker"
        )
        parser.add_argument(
            "--sock-type",
            required=True,
            dest="sock_type",
            type=str,
            choices=["unix", "tcp"],
            help="Socket type the model service worker would use. The options are\n"
            "unix: The model worker expects to unix domain-socket\n"
            "tcp: The model worker expects a host-name and port-number",
        )

        parser.add_argument(
            "--sock-name",
            required=False,
            dest="sock_name",
            type=str,
            help="If 'sock-type' is 'unix', sock-name is expected to be a string. "
            'Eg: --sock-name "test_sock"',
        )

        parser.add_argument(
            "--host",
            type=str,
            help="If 'sock-type' is 'tcp' this is expected to have a host IP address",
        )

        parser.add_argument(
            "--port",
            type=str,
            help="If 'sock-type' is 'tcp' this is expected to have the host port to bind on",
        )

        parser.add_argument(
            "--metrics-config",
            dest="metrics_config",
            type=str,
            help="Metrics configuration file",
        )

        parser.add_argument(
            "--async",
            default=False,
            dest="async_comm",
            action="store_true",
            help="Run async communication worker",
        )

        return parser

    @staticmethod
    def extract_args(args=None):
        parser = ArgParser.ts_parser()
        return parser.parse_args(args) if args else parser.parse_args()
