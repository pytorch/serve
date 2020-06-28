import csv
import json
import os
import shutil
import time
from subprocess import Popen, PIPE

import click
import click_config_file
import matplotlib.pyplot as plt
import pandas as pd

default_ab_params = {'url': "https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar",
                     'device': 'cpu',
                     'exec_env': 'local',
                     'batch_size': 1,
                     'batch_delay': 200,
                     'workers': 4,
                     'concurrency': 10,
                     'requests': 100,
                     'input': '../examples/image_classifier/kitten.jpg',
                     'content_type': 'application/jpg',
                     'image': '',
                     'docker_runtime': ''}

execution_params = default_ab_params
result_file = "/tmp/benchmark/result.txt"
metric_log = "/tmp/benchmark/logs/model_metrics.log"


def json_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        return json.load(config_data)


@click.command()
@click.argument('preset', default='custom')
@click.option('--url', '-u', default='https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar',
              help='input model url')
@click.option('--device', '-d', type=click.Choice(['cpu', 'gpu'], case_sensitive=False), default='cpu',
              help='execution device type')
@click.option('--exec_env', '-e', type=click.Choice(['local', 'docker'], case_sensitive=False), default='local',
              help='execution environment')
@click.option('--concurrency', '-c', default=10, help='concurrency')
@click.option('--requests', '-r', default=100, help='number of requests')
@click.option('--batch_size', '-bs', default=1, help='batch size')
@click.option('--batch_delay', '-bd', default=200, help='batch delay')
@click.option('--input', '-i', 'input_file', default='../examples/image_classifier/kitten.jpg',
              type=click.Path(exists=True),
              help='model input')
@click.option('--workers', '-w', default=4, help='model workers')
@click.option('--content_type', '-ic', default='application/jpg', help='content type')
@click.option('--image', '-di', default='', help='custom docker image')
@click.option('--docker_runtime', '-dr', default='', help='docker runtime')
@click.option('--ts', '-ts', type=click.BOOL, default=False, help='use already running docker instace')
@click_config_file.configuration_option(provider=json_provider, help="read configuration from a JSON file")
def ab_benchmark(preset, url, device, exec_env, concurrency, requests, batch_size, batch_delay, input_file, workers,
                 content_type, image, docker_runtime, ts):
    input_params = {'url': url,
                    'device': device,
                    'exec_env': exec_env,
                    'batch_size': batch_size,
                    'batch_delay': batch_delay,
                    'workers': workers,
                    'concurrency': concurrency,
                    'requests': requests,
                    'input': input_file,
                    'content_type': content_type,
                    'image': image,
                    'docker_runtime': docker_runtime
                    }

    # set ab params
    preset_ab_run[preset]()
    update_exec_params(input_params)
    click.secho("Starting AB benchmark suite..", fg='green')
    click.secho(f"\n\nConfigured execution parameters are:", fg='green')
    click.secho(f"{execution_params}", fg="blue")

    # Setup execution env
    if not ts:
        if execution_params['exec_env'] is 'local':
            click.secho("\n\nSetting up local execution..", fg='green')
            local_torserve_start()
        else:
            click.secho("\n\nSetting up docker execution..", fg='green')
            docker_torchserve_start()
    exec_ab_benchmark()
    report_generate()


def exec_ab_benchmark():
    click.secho("*Registering model...", fg='green')
    RURL = f"?model_name=benchmark&url={execution_params['url']}&batch_size={execution_params['batch_size']}&" \
           f"max_batch_delay={execution_params['batch_delay']}&initial_workers={execution_params['workers']}&synchronous=true"
    execute(f"curl -X POST \"http://localhost:8081/models{RURL}\"", wait=True)

    click.secho("\n\nExecuting Apache Bench tests ...", fg='green')
    click.secho("*Executing inference performance test", fg='green')
    ab_cmd = f"ab -c {execution_params['concurrency']}  -n {execution_params['requests']} -k -p /tmp/benchmark/input -T " \
             f"{execution_params['content_type']} http://127.0.0.1:8080/predictions/benchmark > {result_file}"
    execute(ab_cmd, wait=True)

    click.secho("*Unregistering model ...", fg='green')
    execute("curl -X DELETE \"http://localhost:8081/models/benchmark\"", wait=True)
    click.secho("*Terminating Torchserve instance...", fg='green')
    execute("torchserve --stop", wait=True)
    click.secho("Apache Bench Execution completed.", fg='green')


def execute(command, wait=False, stdout=None, stderr=None, shell=True):
    print(command)
    cmd = Popen(command, shell=shell, close_fds=True, stdout=stdout, stderr=stderr, universal_newlines=True)
    if wait:
        cmd.wait()
    return cmd


def execute_return_stdout(cmd):
    proc = execute(cmd, stdout=PIPE)
    return proc.communicate()[0].strip()


def local_torserve_start():
    click.secho("*Terminating any existing Torchserve instance ...", fg='green')
    execute("torchserve --stop", wait=True)
    click.secho("*Setting up model store...", fg='green')
    resolve_local_exec_dependency(execution_params['input'])
    click.secho("*Starting local Torchserve instance...", fg='green')
    execute("torchserve --start --model-store /tmp/model_store "
            "--ts-config /tmp/benchmark/conf/config.properties > /tmp/benchmark/logs/model_metrics.log")
    time.sleep(3)


def docker_torchserve_start():
    resolve_docker_exec_dependency(execution_params['input'])
    enable_gpu = ''
    if execution_params['image']:
        docker_image = execution_params['image']
    else:
        if execution_params['device'] is 'cpu':
            docker_image = "pytorch/torchserve:latest"
        else:
            docker_image = "pytorch/torchserve:latest-gpu"
            enable_gpu = "--gpus 4"
        execute(f"docker pull {docker_image}", wait=True)

    # delete existing ts conatiner instance
    click.secho("*Removing existing ts conatiner instance", fg='green')
    execute('docker rm -f ts', wait=True)

    click.secho(f"*Starting docker container of image {docker_image}", fg='green')
    docker_run_cmd = f"docker run {execution_params['docker_runtime']} --name ts --user root:root -p 8080:8080 -p 8081:8081 " \
                     f"-v /tmp/benchmark:/tmp/benchmark -itd pytorch/torchserve:latest " \
                     f"\"torchserve --start --ts-config /tmp/benchmark/conf/config.properties > /tmp/benchmark/logs/model_metrics.log\""
    execute(docker_run_cmd, wait=True)
    time.sleep(5)


def resolve_local_exec_dependency(input_file):
    shutil.rmtree('/tmp/model_store/', ignore_errors=True)
    os.makedirs("/tmp/model_store/", exist_ok=True)
    resolve_common_exec_dependency(input_file)


def resolve_docker_exec_dependency(input_file):
    resolve_common_exec_dependency(input_file)


def resolve_common_exec_dependency(input_file):
    input = input_file
    shutil.rmtree("/tmp/benchmark", ignore_errors=True)
    os.makedirs("/tmp/benchmark/conf", exist_ok=True)
    os.makedirs("/tmp/benchmark/logs", exist_ok=True)
    shutil.copy('config.properties', '/tmp/benchmark/conf/')
    shutil.copyfile(input, '/tmp/benchmark/input')


def update_exec_params(input_param):
    for k, v in input_param.items():
        if default_ab_params[k] != input_param[k]:
            execution_params[k] = input_param[k]


def report_generate():
    click.secho("\n\nGenerating Reports", fg='green')
    generate_csv_output()
    generate_latency_graph()


def generate_csv_output():
    click.secho(" Generating CSV output...", fg='green')
    execute(f"grep \"PredictionTime\" {metric_log} |  cut -d\"|\" -f1 | cut -d \":\" -f4 > /tmp/benchmark/predict.txt",
            wait=True)
    batched_requests = execution_params['requests'] / execution_params['batch_size']
    line50 = int(batched_requests / 2)
    line90 = int(batched_requests * 9 / 10)
    line99 = int(batched_requests * 99 / 100)

    artifacts = {}
    artifacts['ts_error'] = execute_return_stdout(
        """grep "Failed requests:" {0} | awk '{1}'""".format(result_file, "{print $NF}"))
    artifacts['ts_rps'] = execute_return_stdout(
        """grep "Requests per second:" {0} | awk '{1}'""".format(result_file, "{print $4}"))
    artifacts['ts_p50'] = execute_return_stdout("""grep "50\%" {0} | awk '{1}'""".format(result_file, "{print $NF}"))
    artifacts['ts_p90'] = execute_return_stdout("""grep "90\%" {0} | awk '{1}'""".format(result_file, "{print $NF}"))
    artifacts['ts_p99'] = execute_return_stdout("""grep "90\%" {0} | awk '{1}'""".format(result_file, "{print $NF}"))
    artifacts['ts_mean'] = execute_return_stdout(
        """grep -E "Time per request:.*mean\)" {0} | awk '{1}'""".format(result_file, "{print $4}"))
    artifacts['ts_error_rate'] = execute_return_stdout(
        """echo "scale=2; 100*{0}/{1}" | bc | awk '{2}'""".format(int(artifacts['ts_error']),
                                                                  execution_params['requests'],
                                                                  "{printf \"%f\", $0}"))

    with open('/tmp/benchmark/predict.txt') as f:
        lines = f.readlines()
        artifacts['model_p50'] = lines[line50].strip()
        artifacts['model_p90'] = lines[line90].strip()
        artifacts['model_p99'] = lines[line99].strip()

    with open('/tmp/benchmark/ab_report.csv', 'w') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(artifacts.keys())
        csvwriter.writerow(artifacts.values())

    return artifacts


def generate_latency_graph():
    click.secho("*Preparing graphs...", fg='green')
    df = pd.read_csv('/tmp/benchmark/predict.txt', header=None, names=['latency'])
    iteration = df.index
    latency = df.latency
    plt.xlabel('Requests ')
    plt.ylabel('Prediction time')
    plt.title('Prediction latency')
    plt.bar(iteration, latency)
    plt.savefig("/tmp/benchmark/predict_latency.png")


# Preset test plans (soak, vgg11_1000r_10c,  vgg11_10000r_100c)
def soak():
    execution_params['requests'] = 100000
    execution_params['concurrency'] = 10


def vgg11_1000r_10c():
    execution_params['url'] = 'https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar'
    execution_params['requests'] = 1000
    execution_params['concurrency'] = 10


def vgg11_10000r_100c():
    execution_params['url'] = 'https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar'
    execution_params['requests'] = 10000
    execution_params['concurrency'] = 100


def custom():
    pass


preset_ab_run = {
    "soak": soak,
    "vgg11_1000r_10c": vgg11_1000r_10c,
    "vgg11_10000r_100c": vgg11_10000r_100c,
    "custom": custom
}

if __name__ == '__main__':
    ab_benchmark()
