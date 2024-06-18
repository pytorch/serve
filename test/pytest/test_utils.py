import glob
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from io import TextIOWrapper
from os import path
from pathlib import Path
from queue import Queue
from subprocess import PIPE, STDOUT, Popen

import orjson
import requests

# To help discover margen modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(REPO_ROOT)
from ts_scripts import marsgen as mg

ROOT_DIR = os.path.join(tempfile.gettempdir(), "workspace")
MODEL_STORE = path.join(ROOT_DIR, "model_store/")
CODEBUILD_WD = path.abspath(path.join(__file__, "../../.."))


class PrintTillTheEnd(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self._queue = queue

    def run(self):
        while True:
            line = self._queue.get()
            if not line:
                break
            print(line.strip())


class Tee(threading.Thread):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader
        self.queue1 = Queue()
        self.queue2 = Queue()

    def run(self):
        for line in self.reader:
            self.queue1.put(line)
            self.queue2.put(line)
        self.queue1.put(None)
        self.queue2.put(None)


def start_torchserve(
    model_store=None,
    snapshot_file=None,
    no_config_snapshots=False,
    gen_mar=True,
    plugin_folder=None,
    disable_token=True,
    models=None,
    model_api_enabled=True,
):
    stop_torchserve()
    crate_mar_file_table()
    cmd = ["torchserve", "--start"]
    model_store = model_store if model_store else MODEL_STORE
    if gen_mar:
        mg.gen_mar(model_store)
    cmd.extend(["--model-store", model_store])
    if plugin_folder:
        cmd.extend(["--plugins-path", plugin_folder])
    if snapshot_file:
        cmd.extend(["--ts-config", snapshot_file])
    if no_config_snapshots:
        cmd.extend(["--no-config-snapshots"])
    if disable_token:
        cmd.append("--disable-token")
    if models:
        cmd.extend(["--models", models])
    if model_api_enabled:
        cmd.extend(["--model-api-enabled"])
    print(cmd)

    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    for line in p.stdout:
        print(line.decode("utf8").strip())
        if "Model server started" in str(line).strip():
            break

    splitter = Tee(TextIOWrapper(p.stdout))
    splitter.start()
    print_thread = PrintTillTheEnd(splitter.queue1)
    print_thread.start()

    return splitter.queue2


def stop_torchserve():
    subprocess.run(["torchserve", "--stop", "--foreground"])


def delete_all_snapshots():
    for f in glob.glob("logs/config/*"):
        os.remove(f)
    assert len(glob.glob("logs/config/*")) == 0


def delete_model_store(model_store=None):
    """Removes all model mar files from model store"""
    model_store = model_store if model_store else MODEL_STORE
    for f in glob.glob(model_store + "/*.mar"):
        os.remove(f)


def torchserve_cleanup():
    stop_torchserve()
    delete_model_store()
    delete_all_snapshots()


def register_model(model_name, url):
    params = (
        ("model_name", model_name),
        ("url", url),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    return register_model_with_params(params)


def register_model_with_params(params):
    response = requests.post("http://localhost:8081/models", params=params)
    return response


def unregister_model(model_name):
    response = requests.delete("http://localhost:8081/models/{}".format(model_name))
    return response


def describe_model(model_name, version):
    response = requests.get(
        "http://localhost:8081/models/{}/{}".format(model_name, version)
    )
    return orjson.loads(response.content)


def delete_mar_file_from_model_store(model_store=None, model_mar=None):
    model_store = (
        model_store
        if (model_store is not None)
        else os.path.join(ROOT_DIR, "model_store")
    )
    if model_mar is not None:
        for f in glob.glob(path.join(model_store, model_mar + "*")):
            os.remove(f)


environment_json = "../postman/environment.json"
mar_file_table = {}


def crate_mar_file_table():
    if not mar_file_table:
        with open(
            os.path.join(os.path.dirname(__file__), *environment_json.split("/")), "rb"
        ) as f:
            env = json.loads(f.read())
        for item in env["values"]:
            if item["key"].startswith("mar_path_"):
                mar_file_table[item["key"]] = item["value"]


def model_archiver_command_builder(
    model_name=None,
    version=None,
    model_file=None,
    serialized_file=None,
    handler=None,
    extra_files=None,
    force=False,
    config_file=None,
    runtime=None,
    archive_format=None,
    requirements_file=None,
    export_path=None,
):
    cmd = "torch-model-archiver"

    if model_name:
        cmd += " --model-name {0}".format(model_name)

    if version:
        cmd += " --version {0}".format(version)

    if model_file:
        cmd += " --model-file {0}".format(model_file)

    if serialized_file:
        cmd += " --serialized-file {0}".format(serialized_file)

    if handler:
        cmd += " --handler {0}".format(handler)

    if extra_files:
        cmd += " --extra-files {0}".format(extra_files)

    if runtime:
        cmd += " --runtime {0}".format(runtime)

    if archive_format:
        cmd += " --archive-format {0}".format(archive_format)

    if requirements_file:
        cmd += " --requirements-file {0}".format(requirements_file)

    if config_file:
        cmd += " --config-file {0}".format(config_file)

    if export_path:
        cmd += " --export-path {0}".format(export_path)
    else:
        cmd += " --export-path {0}".format(MODEL_STORE)

    if force:
        cmd += " --force"

    return cmd


def create_model_artifacts(items: dict, force=False, export_path=None) -> str:
    cmd = model_archiver_command_builder(
        model_name=items.get("model_name"),
        version=items.get("version", "1.0"),
        model_file=items.get("model_file"),
        serialized_file=items.get("serialized_file"),
        handler=items.get("handler"),
        extra_files=items.get("extra_files"),
        force=force,
        config_file=items.get("config_file"),
        runtime=items.get("runtime"),
        archive_format=items.get("archive_format"),
        requirements_file=items.get("requirements_file"),
        export_path=export_path,
    )

    print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")
    try:
        subprocess.check_call(cmd, shell=True)
        if str(items.get("archive_format")) == "no-archive":
            model_artifacts = "{0}".format(items.get("model_name"))
        elif str(items.get("archive_format")) == "tgz":
            model_artifacts = "{0}.tar.gz".format(items.get("model_name"))
        else:
            model_artifacts = "{0}.mar".format(items.get("model_name"))
        print("## {0} is generated.\n".format(model_artifacts))
        return model_artifacts
    except subprocess.CalledProcessError as exc:
        print(
            "## {} creation failed !, error: {}\n".format(items.get("model_name"), exc)
        )
        return None


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def cleanup_model_store(model_store=None):
    # rm -rf $MODEL_STORE_DIR / *
    for f in glob.glob(os.path.join(model_store, "*")):
        os.remove(f)
