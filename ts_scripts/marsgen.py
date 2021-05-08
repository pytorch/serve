import argparse
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)
model_store_dir = os.path.join(REPO_ROOT, "model_store")
os.makedirs(model_store_dir, exist_ok=True)

def generate_mars(mar_config):

    f = open(mar_config, "r")
    models = json.loads(f.read())

    for model in models:
        serialized_file_path = None
        if model.get("serialized_file_remote") and model["serialized_file_remote"]:
            os.chdir(model_store_dir)
            serialized_model_file_url = "https://download.pytorch.org/models/{}".format(model["serialized_file_remote"])
            # Download & create DenseNet Model Archive
            urllib.request.urlretrieve(serialized_model_file_url, model["serialized_file_remote"])
            serialized_file_path = os.path.join(model_store_dir, model["serialized_file_remote"])
        elif model.get("serialized_file_local") and model["serialized_file_local"]:
            serialized_file_path = model["serialized_file_local"]

        handler = None
        if model.get("handler") and model["handler"]:
            handler = model["handler"]

        extra_files = None
        if model.get("extra_files") and model["extra_files"]:
            extra_files = model["extra_files"]

        runtime = None
        if model.get("runtime") and model["runtime"]:
            runtime = model["runtime"]

        archive_format = None
        if model.get("archive_format") and model["archive_format"]:
            archive_format = model["archive_format"]

        requirements_file = None
        if model.get("requirements_file") and model["requirements_file"]:
            requirements_file = model["requirements_file"]

        os.chdir(REPO_ROOT)
        export_path = model_store_dir
        if model.get("export_path") and model["export_path"]:
            export_path = model["export_path"]



        cmd = model_archiver_command_builder(model["model_name"], model["version"], model["model_file"],
                                             serialized_file_path, handler, extra_files,
                                             runtime, archive_format, requirements_file, export_path)
        print(f"## In directory: {os.getcwd()} | Executing command: {cmd}")
        sys_exit_code = os.system(cmd)
        if model.get("serialized_file_remote") and model["serialized_file_remote"]:
            os.remove(serialized_file_path)

        if sys_exit_code != 0:
            sys.exit("## {} creation failed !\n".format(model["model_name"]))
        else :
            print("## {}.mar is generated.\n".format(model["model_name"]))

    f.close()

def model_archiver_command_builder(model_name=None, version=None, model_file=None,
                                   serialized_file=None, handler=None, extra_files=None,
                                   runtime=None, archive_format=None, requirements_file=None,
                                   export_path=None, force=True):
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

    if export_path:
        cmd += " --export-path {0}".format(export_path)

    if force:
        cmd += " --force"

    return cmd

if __name__ == "__main__":
    # cmd:
    # python ts_scripts/marsgen.py
    # python ts_scripts/marsgen.py --config my_mar_config.json

    mar_config_file_path = os.path.join(REPO_ROOT, "ts_scripts", "mar_config.json")
    parser = argparse.ArgumentParser(description="Generate model mar files")
    parser.add_argument('--config', default=mar_config_file_path, help="mar file configuration json file")

    args = parser.parse_args()
    generate_mars(args.config)
