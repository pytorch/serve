"""
File to define the entry point to Model Server
"""

import os
import re
import subprocess
import sys
import tempfile
from builtins import str

import psutil

from mms.arg_parser import ArgParser


def start():
    """
    This is the entry point for model server
    :return:
    """
    args = ArgParser.mms_parser().parse_args()
    pid_file = os.path.join(tempfile.gettempdir(), ".model_server.pid")
    pid = None
    if os.path.isfile(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.readline())

    # pylint: disable=too-many-nested-blocks
    if args.stop:
        if pid is None:
            print("Model server is not currently running.")
        else:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                print("Model server stopped.")
            except (OSError, psutil.Error):
                print("Model server already stopped.")
            os.remove(pid_file)
    else:
        if pid is not None:
            try:
                psutil.Process(pid)
                print("Model server is already running, please use mxnet-model-server --stop to stop MMS.")
                exit(1)
            except psutil.Error:
                print("Removing orphan pid file.")
                os.remove(pid_file)

        java_home = os.environ.get("JAVA_HOME")
        java = "java" if not java_home else "{}/bin/java".format(java_home)

        mms_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = [java, "-Dmodel_server_home={}".format(mms_home)]
        if args.log_config:
            log_config = os.path.realpath(args.log_config)
            if not os.path.isfile(log_config):
                print("--log-config file not found: {}".format(log_config))
                exit(1)

            cmd.append("-Dlog4j.configuration=file://{}".format(log_config))

        tmp_dir = os.environ.get("TEMP")
        if tmp_dir:
            if not os.path.isdir(tmp_dir):
                print("Invalid temp directory: {}, please check TEMP environment variable.".format(tmp_dir))
                exit(1)

            cmd.append("-Djava.io.tmpdir={}".format(tmp_dir))

        mms_config = args.mms_config
        mms_conf_file = None
        if mms_config:
            if mms_config == "sagemaker":
                mms_config = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "configs", "sagemaker_config.properties")
            if not os.path.isfile(mms_config):
                print("--mms-config file not found: {}".format(mms_config))
                exit(1)
            mms_conf_file = mms_config
        else:
            mms_conf_file = "config.properties"

        class_path = \
            ".:{}".format(os.path.join(mms_home, "mms/frontend/*"))

        if os.path.isfile(mms_conf_file):
            props = load_properties(mms_conf_file)
            vm_args = props.get("vmargs")
            if vm_args:
                arg_list = vm_args.split()
                if args.log_config:
                    for word in arg_list[:]:
                        if word.startswith("-Dlog4j.configuration="):
                            arg_list.remove(word)
                cmd.extend(arg_list)
            plugins = props.get("plugins_path", None)
            if plugins:
                class_path += ":" + plugins + "/*" if "*" not in plugins else ":" + plugins

        cmd.append("-cp")
        cmd.append(class_path)

        cmd.append("com.amazonaws.ml.mms.ModelServer")

        # model-server.jar command line parameters
        cmd.append("--python")
        cmd.append(sys.executable)

        if mms_conf_file is not None:
            cmd.append("-f")
            cmd.append(mms_conf_file)

        if args.model_store:
            if not os.path.isdir(args.model_store):
                print("--model-store directory not found: {}".format(args.model_store))
                exit(1)

            cmd.append("-s")
            cmd.append(args.model_store)

        if args.models:
            cmd.append("-m")
            cmd.extend(args.models)
            if not args.model_store:
                pattern = re.compile(r"(.+=)?http(s)?://.+", re.IGNORECASE)
                for model_url in args.models:
                    if not pattern.match(model_url) and model_url != "ALL":
                        print("--model-store is required to load model locally.")
                        exit(1)

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


def load_properties(file_path):
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
