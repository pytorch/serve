import argparse
import copy
import datetime
import json
import os
import ruamel.yaml
import shutil
from subprocess import Popen
from ts_scripts import print_env_info
from benchmarks.utils import gen_model_config_json
from benchmarks.utils import gen_metrics_json
from benchmarks.utils import gen_md_report

MODEL_JSON_CONFIG_PATH = './model_json_config'
BENCHMARK_TMP_PATH = '/tmp/benchmark'
BENCHMARK_REPORT_PATH = '/tmp/ts_benchmark'

class BenchmarkConfig:
    def __init__(self, yaml_dict):
        self.yaml_dict = yaml_dict
        self.bm_config = {}
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        self.bm_config["version"] = \
            "torchserve-nightly=={}.{}.{}".format(yesterday.year, yesterday.month, yesterday.day)
        self.bm_config["hardware"] = 'cpu'

    def ts_version(self, version):
        for k, v in version.items():
            if k == "branch":
                self.bm_config["version"] = v
            elif k == "nightly":
                self.bm_config["version"] = "torchserve-nightly=={}".format(v)
            elif k == "release":
                self.bm_config["version"] = "torchserve=={}".format(v)
            break

    def models(self, model_files):
        self.bm_config["models"] = model_files

    def hardware(self, hw):
        self.bm_config["hardware"] = hw

    def metrics_cmd(self, cmd):
        cmd_options = []
        for key_value in cmd:
            for k, v in key_value.items():
                if k == "cmd":
                    cmd_options.append(v)
                elif k == '--namespace':
                    cmd_options.append(k)
                    cmd_options.append(''.join(v))
                else:
                    cmd_options.append(k)
                    cmd_options.append(v)
                break

        self.bm_config["metrics_cmd"] = ' '.join(cmd_options)

    def report_cmd(self, cmd):
        cmd_options = []
        for key_value in cmd:
            for k, v in key_value.items():
                if k == "cmd":
                    cmd_options.append(v)
                elif k == "dest":
                    for i in range(len(v)):
                        if v[i] == "today()":
                            today = datetime.date.today()
                            v[i] = "{}-{}-{}".format(today.year, today.month, today.day)
                            break
                    cmd_options.append('{}/{}'.format('/'.join(v), self.bm_config["version"]))
                else:
                    cmd_options.append(v)
                break

        self.bm_config["report_cmd"] = ' '.join(cmd_options)

    def load_config(self):
        report_cmd = None
        for k, v in self.yaml_dict.items():
            if k == "ts_version":
                self.ts_version(v)
            elif k == "models":
                self.models(v)
            elif k == "hardware":
                self.hardware(v)
            elif k == "metrics_cmd":
                self.metrics_cmd(v)
            elif k == "report_cmd":
                report_cmd = v


        self.bm_config["models_config_path"] = \
            './{}/cpu'.format(MODEL_JSON_CONFIG_PATH) \
                if self.bm_config["hardware"] == 'cpu' \
                else './{}}/gpu'.format(MODEL_JSON_CONFIG_PATH)

        if report_cmd:
            self.report_cmd(report_cmd)

        for k, v in self.bm_config.items():
            print("{}={}".format(k, v))

def load_benchmark_config(bm_config_path, skip_ts_install):
    yaml = ruamel.yaml.YAML()
    with open(bm_config_path, 'r') as f:
        yaml_dict = yaml.load(f)

        benchmark_config = BenchmarkConfig(yaml_dict)
        benchmark_config.load_config()

    if skip_ts_install:
        benchmark_config["version"] = print_env_info.get_torchserve_version()

    return benchmark_config.bm_config

def benchmark_env_setup(bm_config, skip_ts_install):
    install_torchserve(skip_ts_install, bm_config["hardware"], bm_config["version"])
    setup_benchmark_path(bm_config["model_config_path"])
    build_model_json_config(bm_config["models"])

def install_torchserve(skip_ts_install, hw, ts_version):
    cmd = 'pip install -r benchmarks/requirements-ab.txt'
    execute(cmd, wait=True)

    if skip_ts_install:
        return

    # git checkout branch if it is needed
    cmd = 'git checkout master && git reset --hard && git clean -dffx . && git pull --rebase'
    execute(cmd, wait=True)

    ts_install_cmd = None
    if ts_version.startswith("torchserve==") or ts_version.startswith("torchserve-nightly=="):
        ts_install_cmd = 'pip install {}'.format(ts_version)
    else:
        cmd = 'git checkout {}'.format(ts_version)
        execute(cmd, wait=True)

    # install_dependencies.py
    if hw == 'gpu':
        cmd = 'python ts_scripts/install_dependencies.py --environment dev --cuda cu102'
    else:
        cmd = 'python ts_scripts/install_dependencies.py --environment dev'
    execute(cmd, wait=True)

    # install torchserve
    if ts_install_cmd is None:
        ts_install_cmd = 'python ts_scripts/install_from_src.py'
    execute(ts_install_cmd, wait=True)

def setup_benchmark_path(model_config_path):
    benchmark_path_list = [BENCHMARK_TMP_PATH, BENCHMARK_REPORT_PATH, model_config_path]
    try:
        for benchmark_path in benchmark_path_list:
            os.rmdir(benchmark_path)
            os.mkdir(benchmark_path)
    except OSError as e:
        print("Error: %s : %s" % (benchmark_path, e.strerror))

def build_model_json_config(models):
    for model in models:
        input_file = './benchmarks/models_config/{}'.format(model)
        gen_model_config_json.convert_yaml_to_json(input_file, MODEL_JSON_CONFIG_PATH)

def run_benchmark(bm_config):
    for model_json_config in os.listdir(bm_config["models_config_path"]):
        if model_json_config.endswith(".json"):
            # call benchmark-ab.py
            os.rmdir('./logs')
            cmd = 'python ./benchmarks/benchmark-ab.py --config_properties ' \
                  './benchmarks/config.properties --config {}/{}'\
                .format(bm_config["models_config_path"], model_json_config)
            execute(cmd, wait=True)

            # generate stats metrics from ab_report.csv
            bm_model = model_json_config[0: -len('.json')]
            print('bm_model={}'.format(bm_model))

            gen_metrics_json.gen_metric(
                '{}/ab_report.csv'.format(BENCHMARK_TMP_PATH),
                '{}/logs/stats_metrics.json'.format(BENCHMARK_TMP_PATH)
            )

            # load stats metrics to remote metrics storage
            if "metrics_cmd" in bm_config:
                execute(bm_config["metrics_cmd"], wait=True)

            # cp benchmark logs to local
            bm_model_log_path = '{}/{}'.format(BENCHMARK_REPORT_PATH, bm_model)
            os.mkdir(bm_model_log_path)
            csv_file = '{}/ab_report.csv'.format(BENCHMARK_TMP_PATH)
            if os.path.exists(csv_file):
                shutil.move('{}/ab_report.csv'.format(csv_file), bm_model_log_path)
            cmd = 'tar cvzf {}/benchmark.tar.gz {}'.format(bm_model_log_path, BENCHMARK_TMP_PATH)
            execute(cmd, wait=True)

            cmd = 'tar cvzf {}/logs.tar.gz ./logs'.format(BENCHMARK_REPORT_PATH)
            execute(cmd, wait=True)

    # generate final report
    gen_md_report.iterate_subdir(
        BENCHMARK_REPORT_PATH,
        '{}/repord.md'.format(BENCHMARK_REPORT_PATH),
        bm_config["hardware"],
        bm_config["version"])

    # load logs to remote storage
    if "report_cmd" in bm_config:
        execute(bm_config["report_cmd"], wait=True)

def clean_up_benchmark_env(bm_config):
    os.rmdir(BENCHMARK_TMP_PATH)
    os.rmdir(MODEL_JSON_CONFIG_PATH)

def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(command, shell=shell, close_fds=True, stdout=stdout, stderr=stderr, universal_newlines=True)
    if wait:
        cmd.wait()
    return cmd

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="benchmark config yaml file path",
    )
    parser.add_argument(
        "--skip",
        action="store",
        help="skip torchserve installation",
    )

    arguments = parser.parse_args()
    bm_config = load_benchmark_config(arguments.input, arguments.skip)
    benchmark_env_setup(bm_config, arguments.skip)
    run_benchmark(bm_config)
    print("benchmark_serving.sh finished successfully.")

if __name__ == "__main__":
    main()