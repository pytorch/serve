# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This module parses the arguments given through the mxnet-model-server command-line. This is used by model-server
at runtime.
"""

import argparse


# noinspection PyTypeChecker
class ArgParser(object):
    """
    Argument parser for mxnet-model-server and mxnet-model-export commands
    More detailed example is available at https://github.com/awslabs/mxnet-model-server/blob/master/README.md
    """
    @staticmethod
    def mms_parser():
        """
        Argument parser for mxnet-model-server start service
        """
        parser = argparse.ArgumentParser(prog='mxnet-model-server', description='MXNet Model Server')

        sub_parse = parser.add_mutually_exclusive_group(required=False)
        sub_parse.add_argument('--start', action='store_true', help='Start the model-server')
        sub_parse.add_argument('--stop', action='store_true', help='Stop the model-server')

        parser.add_argument('--mms-config',
                            dest='mms_config',
                            help='Configuration file for model server')
        parser.add_argument('--model-store',
                            dest='model_store',
                            help='Model store location where models can be loaded')
        parser.add_argument('--models',
                            metavar='MODEL_PATH1 MODEL_NAME=MODEL_PATH2...',
                            nargs='+',
                            help='Models to be loaded using [model_name=]model_location format. '
                                 'Location can be a HTTP URL, a model archive file or directory '
                                 'contains model archive files in MODEL_STORE.')
        parser.add_argument('--log-config',
                            dest='log_config',
                            help='Log4j configuration file for model server')
        parser.add_argument('--foreground',
                            help='Run the model server in foreground. If this option is disabled, the model server'
                                 ' will run in the background.',
                            action='store_true')

        return parser

    @staticmethod
    def model_service_worker_args():
        """
        ArgParser for backend worker. Takes the socket name and socket type.
        :return:
        """
        parser = argparse.ArgumentParser(prog='model-server-worker', description='Model Server Worker')
        parser.add_argument('--sock-type',
                            required=True,
                            dest="sock_type",
                            type=str,
                            choices=["unix", "tcp"],
                            help='Socket type the model service worker would use. The options are\n'
                                 'unix: The model worker expects to unix domain-socket\n'
                                 'tcp: The model worker expects a host-name and port-number')

        parser.add_argument('--sock-name',
                            required=False,
                            dest="sock_name",
                            type=str,
                            help='If \'sock-type\' is \'unix\', sock-name is expected to be a string. '
                                 'Eg: --sock-name \"test_sock\"')

        parser.add_argument('--host',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have a host IP address')

        parser.add_argument('--port',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have the host port to bind on')

        return parser

    @staticmethod
    def extract_args(args=None):
        parser = ArgParser.mms_parser()
        return parser.parse_args(args) if args else parser.parse_args()
