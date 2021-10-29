import argparse
import json
import os
import sys
import urllib.request
import shutil
import subprocess

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)
MODEL_STORE_DIR = os.path.join(REPO_ROOT, "model_store_gen")
os.makedirs(MODEL_STORE_DIR, exist_ok=True)
MAR_CONFIG_FILE_PATH = os.path.join(REPO_ROOT, "ts_scripts", "mar_config.json")

def delete_model_store_gen_dir():
    print(f"## Deleting model_store_gen_dir: {MODEL_STORE_DIR}\n")
    mar_set.clear()
    if os.path.exists(MODEL_STORE_DIR):
        try:
            shutil.rmtree(MODEL_STORE_DIR)
        except OSError as e:
            print("Error: %s : %s" % (MODEL_STORE_DIR, e.strerror))

mar_set = set()
def gen_mar(model_store=None):
    print(f"## Starting gen_mar: {model_store}\n")
    if len(mar_set) == 0:
        generate_mars(mar_config=MAR_CONFIG_FILE_PATH, model_store_dir=MODEL_STORE_DIR)

    if model_store is not None and os.path.exists(model_store):
        print("## Create symlink for mar files\n")
        for mar_file in mar_set:
            src = f"{MODEL_STORE_DIR}/{mar_file}"
            dst = f"{model_store}/{mar_file}"
            if os.path.exists(dst):
                print(f"## {dst} already exists.\n")
            else:
                os.symlink(src, dst)
                print(f"## Symlink {src}, {dst} successfully.")


def generate_mars(mar_config=MAR_CONFIG_FILE_PATH, model_store_dir=MODEL_STORE_DIR):
    """
    By default generate_mars reads ts_scripts/mar_config.json and outputs mar files in dir model_store_gen
    - mar_config.json defines a list of models' mar file parameters. They are:
    - "model_name": model name
    - "version": model version
    - "model_file": the path of file model.py
    - "serialized_file_remote": the url of file .pth or .pt
    - "serialized_file_local": the path of file .pth or .pt
    - "gen_scripted_file_path": the python script path of building .pt file
    - "handler": handler can be either default handler or handler path
    - "extra_files": the paths of extra files
    Note: To generate .pt file, "serialized_file_remote" and "gen_scripted_file_path" must be provided
    """
    print(f"## Starting generate_mars, mar_config:{mar_config}, model_store_dir:{model_store_dir}\n")
    mar_set.clear()
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    with open(mar_config) as f:
        models = json.loads(f.read())

        for model in models:
            serialized_file_path = None
            if model.get("serialized_file_remote") and model["serialized_file_remote"]:
                if model.get("gen_scripted_file_path") and model["gen_scripted_file_path"]:
                    subprocess.run(["python", model["gen_scripted_file_path"]])
                else:
                    serialized_model_file_url = \
                        "https://download.pytorch.org/models/{}".format(model["serialized_file_remote"])
                    urllib.request.urlretrieve(
                        serialized_model_file_url,
                        f'{model_store_dir}/{model["serialized_file_remote"]}')
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

            export_path = model_store_dir
            if model.get("export_path") and model["export_path"]:
                export_path = model["export_path"]

            cmd = model_archiver_command_builder(model["model_name"], model["version"], model["model_file"],
                                                 serialized_file_path, handler, extra_files,
                                                 runtime, archive_format, requirements_file, export_path)
            print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")
            try:
                subprocess.check_call(cmd, shell=True)
                marfile = "{}.mar".format(model["model_name"])
                print("## {} is generated.\n".format(marfile))
                mar_set.add(marfile)
            except subprocess.CalledProcessError as exc:
                print("## {} creation failed !, error: {}\n".format(model["model_name"], exc))

            if model.get("serialized_file_remote") and \
                    model["serialized_file_remote"] and \
                    os.path.exists(serialized_file_path):
                os.remove(serialized_file_path)
    os.chdir(cwd)


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

    parser = argparse.ArgumentParser(description="Generate model mar files")
    parser.add_argument('--config', default=MAR_CONFIG_FILE_PATH, help="mar file configuration json file")
    parser.add_argument('--model-store', default=MODEL_STORE_DIR, help="model store dir")

    args = parser.parse_args()
    generate_mars(args.config, MODEL_STORE_DIR)
