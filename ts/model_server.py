"""
File to define the entry point to Model Server
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
from builtins import str
from typing import Dict

import psutil

from ts.arg_parser import ArgParser
from ts.version import __version__

TS_NAMESPACE = "org.pytorch.serve.ModelServer"


def start() -> None:
    """
    This is the entry point for model server
    :return:
    """
    args = ArgParser.ts_parser().parse_args()
    pid_file = os.path.join(tempfile.gettempdir(), ".model_server.pid")
    pid = None
    if os.path.isfile(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.readline())
            try:
                process_from_pid_file = psutil.Process(pid)
                pid = pid if TS_NAMESPACE in process_from_pid_file.cmdline() else None
            except psutil.Error:
                pid = None
                print("Removing orphan pid file.")
                os.remove(pid_file)
    # pylint: disable=too-many-nested-blocks
    if args.version:
        print("TorchServe Version is {}".format(__version__))
        return
    if args.stop:
        if pid is None:
            print("TorchServe is not currently running.")
        else:
            try:
                parent = psutil.Process(pid)
                parent.terminate()
                if args.foreground:
                    try:
                        parent.wait(timeout=60)
                    except psutil.TimeoutExpired:
                        print("Stopping TorchServe took too long.")
                else:
                    print("TorchServe has stopped.")
            except (OSError, psutil.Error):
                print("TorchServe already stopped.")
            os.remove(pid_file)
    else:
        if pid is not None:
            try:
                psutil.Process(pid)
                print(
                    "TorchServe is already running, please use torchserve --stop to stop TorchServe."
                )
                sys.exit(1)
            except psutil.Error:
                print("Removing orphan pid file.")
                os.remove(pid_file)

        java_home = os.environ.get("JAVA_HOME")
        java = "java" if not java_home else "{}/bin/java".format(java_home)

        ts_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = [java, "-Dmodel_server_home={}".format(ts_home)]
        if args.log_config:
            log_config = os.path.realpath(args.log_config)
            if not os.path.isfile(log_config):
                print("--log-config file not found: {}".format(log_config))
                sys.exit(1)

            cmd.append("-Dlog4j.configurationFile=file://{}".format(log_config))

        tmp_dir = os.environ.get("TEMP")
        if tmp_dir:
            if not os.path.isdir(tmp_dir):
                print(
                    "Invalid temp directory: {}, please check TEMP environment variable.".format(
                        tmp_dir
                    )
                )
                sys.exit(1)

            cmd.append("-Djava.io.tmpdir={}".format(tmp_dir))

        ts_config = os.environ.get("TS_CONFIG_FILE")
        if ts_config is None:
            ts_config = args.ts_config
        ts_conf_file = None
        if ts_config:
            if not os.path.isfile(ts_config):
                print("--ts-config file not found: {}".format(ts_config))
                sys.exit(1)
            ts_conf_file = ts_config

        platform_path_separator = {"Windows": "", "Darwin": ".:", "Linux": ".:"}
        class_path = "{}{}".format(
            platform_path_separator[platform.system()],
            os.path.join(ts_home, "ts", "frontend", "*"),
        )

        if ts_conf_file and os.path.isfile(ts_conf_file):
            props = load_properties(ts_conf_file)
            vm_args = props.get("vmargs")
            if vm_args:
                print(
                    "Warning: TorchServe is using non-default JVM parameters: {}".format(
                        vm_args
                    )
                )
                arg_list = vm_args.split()
                if args.log_config:
                    for word in arg_list[:]:
                        if word.startswith("-Dlog4j.configurationFile="):
                            arg_list.remove(word)
                cmd.extend(arg_list)
            plugins = props.get("plugins_path", None)
            if plugins:
                class_path += (
                    ":" + plugins + "/*" if "*" not in plugins else ":" + plugins
                )

            if not args.model_store and props.get("model_store"):
                args.model_store = props.get("model_store")

        if args.plugins_path:
            class_path += (
                ":" + args.plugins_path + "/*"
                if "*" not in args.plugins_path
                else ":" + args.plugins_path
            )

        cmd.append("-cp")
        cmd.append(class_path)

        cmd.append("org.pytorch.serve.ModelServer")

        # model-server.jar command line parameters
        cmd.append("--python")
        cmd.append(sys.executable)

        if ts_conf_file is not None:
            cmd.append("-f")
            cmd.append(ts_conf_file)

        if args.model_store:
            if not os.path.isdir(args.model_store):
                print("--model-store directory not found: {}".format(args.model_store))
                sys.exit(1)

            cmd.append("-s")
            cmd.append(args.model_store)
        else:
            print("Missing mandatory parameter --model-store")
            sys.exit(1)

        if args.workflow_store:
            if not os.path.isdir(args.workflow_store):
                print(
                    "--workflow-store directory not found: {}".format(
                        args.workflow_store
                    )
                )
                sys.exit(1)

            cmd.append("-w")
            cmd.append(args.workflow_store)

        if args.no_config_snapshots:
            cmd.append("-ncs")

        if args.models:
            cmd.append("-m")
            cmd.extend(args.models)
            if not args.model_store:
                pattern = re.compile(r"(.+=)?http(s)?://.+", re.IGNORECASE)
                for model_url in args.models:
                    if not pattern.match(model_url) and model_url != "ALL":
                        print("--model-store is required to load model locally.")
                        sys.exit(1)

        try:
            process = subprocess.Popen(cmd)
            pid = process.pid
            with open(pid_file, "w") as pf:
                pf.write(str(pid))
            if args.foreground:
                process.wait()
        except OSError as e:
            if e.errno == 2:
                print("java not found, please make sure JAVA_HOME is set properly.")
            else:
                print("start java frontend failed:", sys.exc_info())


def load_properties(file_path: str) -> Dict[str, str]:
    """
    Read properties file into map.
    """
    props = {}
    with open(file_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                pair = line.split("=", 1)
                if len(pair) > 1:
                    key = pair[0].strip()
                    props[key] = pair[1].strip()
    return props


if __name__ == "__main__":
    start()
