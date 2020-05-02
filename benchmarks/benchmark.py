#!/usr/bin/env python3

"""
Execute the TorchServe Benchmark.  For instructions, run with the --help flag
"""

# pylint: disable=redefined-builtin

import argparse
import itertools
import multiprocessing
import os
import pprint
import shutil
import subprocess
import sys
import time
import traceback
from functools import reduce
from urllib.request import urlretrieve

import pandas as pd

BENCHMARK_DIR = "/tmp/TSBenchmark/"

OUT_DIR = os.path.join(BENCHMARK_DIR, 'out/')
RESOURCE_DIR = os.path.join(BENCHMARK_DIR, 'resource/')

RESOURCE_MAP = {
    'kitten.jpg': 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
}

# Listing out all the JMX files
JMX_IMAGE_INPUT_MODEL_PLAN = 'imageInputModelPlan.jmx'
JMX_PING_PLAN = 'pingPlan.jmx'
JMX_CONCURRENT_LOAD_PLAN = 'concurrentLoadPlan.jmx'
JMX_CONCURRENT_SCALE_CALLS = 'concurrentScaleCalls.jmx'
JMX_MULTIPLE_MODELS_LOAD_PLAN = 'multipleModelsLoadPlan.jmx'
JMX_GRAPHS_GENERATOR_PLAN = 'graphsGenerator.jmx'

# Listing out the models tested
MODEL_RESNET_18 = 'resnet-18'
MODEL_SQUEEZE_NET = 'squeezenet'

MODEL_MAP = {
    MODEL_SQUEEZE_NET: (JMX_IMAGE_INPUT_MODEL_PLAN, {'url': 'https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar', 'model_name': MODEL_SQUEEZE_NET, 'input_filepath': 'kitten.jpg'}),
    MODEL_RESNET_18: (JMX_IMAGE_INPUT_MODEL_PLAN, {'url': 'https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar', 'model_name': MODEL_RESNET_18, 'input_filepath': 'kitten.jpg'}),
}


# Mapping of which row is relevant for a given JMX Test Plan
EXPERIMENT_RESULTS_MAP = {
    JMX_IMAGE_INPUT_MODEL_PLAN: ['Inference Request'],
    JMX_PING_PLAN: ['Ping Request'],
    JMX_CONCURRENT_LOAD_PLAN: ['Load Model Request'],
    JMX_CONCURRENT_SCALE_CALLS: ['Scale Up Model', 'Scale Down Model'],
    JMX_MULTIPLE_MODELS_LOAD_PLAN: ['Inference Request']
}


JMETER_RESULT_SETTINGS = {
    'jmeter.reportgenerator.overall_granularity': 1000,
    # 'jmeter.reportgenerator.report_title': '"TorchServe Benchmark Report Dashboard"',
    'aggregate_rpt_pct1': 50,
    'aggregate_rpt_pct2': 90,
    'aggregate_rpt_pct3': 99,
}

# Dictionary of what's present in the output csv generated v/s what we want to change the column name to for readability
AGGREGATE_REPORT_CSV_LABELS_MAP = {
    'aggregate_report_rate': 'Throughput',
    'average': 'Average',
    'aggregate_report_median': 'Median',
    'aggregate_report_90%_line': 'aggregate_report_90_line',
    'aggregate_report_99%_line': 'aggregate_report_99_line',
    'aggregate_report_error%': 'aggregate_report_error'
}


CELLAR = '/home/linuxbrew/.linuxbrew/Homebrew/Cellar/jmeter/' if 'linux' in sys.platform else '/usr/local/Cellar/jmeter'
JMETER_VERSION = os.listdir(CELLAR)[0]
CMDRUNNER = '{}/{}/libexec/lib/cmdrunner-2.2.jar'.format(CELLAR, JMETER_VERSION)
JMETER = '{}/{}/libexec/bin/jmeter'.format(CELLAR, JMETER_VERSION)
TS_BASE = reduce(lambda val, func: func(val), (os.path.abspath(__file__),) + (os.path.dirname,) * 2)
JMX_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jmx')
CONFIG_PROP = os.path.join(TS_BASE, 'benchmarks', 'config.properties')
CONFIG_PROP_TEMPLATE = os.path.join(TS_BASE, 'benchmarks', 'config_template.properties')

DOCKER_TS_BASE = "/serve"
DOCKER_CONFIG_PROP = os.path.join(DOCKER_TS_BASE, 'benchmarks', 'config.properties')

ALL_BENCHMARKS = list(itertools.product(('latency', 'throughput'), (MODEL_RESNET_18)))


BENCHMARK_NAMES = ['latency', 'throughput']

class ChDir:
    def __init__(self, path):
        self.curPath = os.getcwd()
        self.path = path

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.curPath)


def basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_resource(name):
    url = RESOURCE_MAP[name]
    path = os.path.join(RESOURCE_DIR, name)
    if not os.path.exists(path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        urlretrieve(url, path)
    return path


def run_process(cmd, wait=True, **kwargs):
    output = None if pargs.verbose else subprocess.DEVNULL
    if pargs.verbose:
        print(' '.join(cmd) if isinstance(cmd, list) else cmd)
    if not kwargs.get('shell') and isinstance(cmd, str):
        cmd = cmd.split(' ')
    if 'stdout' not in kwargs:
        kwargs['stdout'] = output
    if 'stderr' not in kwargs:
        kwargs['stderr'] = output
    p = subprocess.Popen(cmd, **kwargs)
    if wait:
        p.wait()
    return p


def run_single_benchmark(jmx, jmeter_args=dict(), threads=100, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(OUT_DIR, benchmark_name, basename(benchmark_model))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    protocol = 'http'
    hostname = '127.0.0.1'
    port = 8080
    threads = pargs.threads[0] if pargs.threads else threads
    workers = pargs.workers[0] if pargs.workers else (
        pargs.gpus[0] if pargs.gpus else multiprocessing.cpu_count()
    )

    if pargs.ts:
        url = pargs.ts[0]
        if '://' in url:
            protocol, url = url.split('://')
        if ':' in url:
            hostname, port = url.split(':')
            port = int(port)
        else:
            hostname = url
            port = 80
    else:
        # Start TorchServe
        '''
        Default docker files do not exist and same needs to be added.
        '''
        raise Exception('Docker not supported at this moment. Use --ts switch.')

        docker = 'nvidia-docker' if pargs.gpus else 'docker'
        container = 'ts_benchmark_gpu' if pargs.gpus else 'ts_benchmark_cpu'
        docker_path = 'torchserve-model-server:nightly-torch-gpu' \
            if pargs.gpus else 'torchserve-model-server:nightly-torch-cpu'
        if pargs.docker:
            container = 'ts_benchmark_{}'.format(pargs.docker[0].split('/')[1])
            docker_path = pargs.docker[0]
        run_process("{} rm -f {}".format(docker, container))
        docker_run_call = "{} run --name {} -p 8080:8080 -p 8081:8081 -itd {}".format(docker, container, docker_path)
        run_process(docker_run_call)

    management_port = int(pargs.management[0]) if pargs.management else port + 1
    time.sleep(300)

    try:
        # temp files
        tmpfile = os.path.join(out_dir, 'output.jtl')
        logfile = os.path.join(out_dir, 'jmeter.log')
        outfile = os.path.join(out_dir, 'out.csv')
        perfmon_file = os.path.join(out_dir, 'perfmon.csv')
        graphsDir = os.path.join(out_dir, 'graphs')
        reportDir = os.path.join(out_dir, 'report')

        # run jmeter
        run_jmeter_args = {
            'hostname': hostname,
            'port': port,
            'management_port': management_port,
            'protocol': protocol,
            'min_workers': workers,
            'rampup': 5,
            'threads': threads,
            'loops': int(pargs.loops[0]),
            'perfmon_file': perfmon_file
        }
        run_jmeter_args.update(JMETER_RESULT_SETTINGS)
        run_jmeter_args.update(jmeter_args)
        run_jmeter_args.update(dict(zip(pargs.options[::2], pargs.options[1::2])))
        abs_jmx = jmx if os.path.isabs(jmx) else os.path.join(JMX_BASE, jmx)
        jmeter_args_str = ' '.join(sorted(['-J{}={}'.format(key, val) for key, val in run_jmeter_args.items()]))
        jmeter_call = '{} -n -t {} {} -l {} -j {} -e -o {}'.format(JMETER, abs_jmx, jmeter_args_str, tmpfile, logfile, reportDir)
        run_process(jmeter_call)
        print('Processing jmeter output')
        time.sleep(30)
        # run AggregateReport
        ag_call = 'java -jar {} --tool Reporter --generate-csv {} --input-jtl {} --plugin-type AggregateReport'.format(CMDRUNNER, outfile, tmpfile)
        run_process(ag_call)

        # Generate output graphs
        gLogfile = os.path.join(out_dir, 'graph_jmeter.log')
        graphing_args = {
            'raw_output': graphsDir,
            'jtl_input': tmpfile
        }
        graphing_args.update(JMETER_RESULT_SETTINGS)
        gjmx = os.path.join(JMX_BASE, JMX_GRAPHS_GENERATOR_PLAN)
        graphing_args_str = ' '.join(['-J{}={}'.format(key, val) for key, val in graphing_args.items()])
        graphing_call = '{} -n -t {} {} -j {}'.format(JMETER, gjmx, graphing_args_str, gLogfile)
        run_process(graphing_call)

        print("Output available at {}".format(out_dir))
        print("Report generated at {}".format(os.path.join(reportDir, 'index.html')))

        data_frame = pd.read_csv(outfile, index_col=0)
        report = list()
        for val in EXPERIMENT_RESULTS_MAP[jmx]:
            for full_val in [fv for fv in data_frame.index if val in fv]:
                report.append(decorate_metrics(data_frame, full_val))

        return report

    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()


def run_multi_benchmark(key, xs, *args, **kwargs):
    out_dir = os.path.join(OUT_DIR, benchmark_name, basename(benchmark_model))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    reports = dict()
    out_dirs = []
    for i, x in enumerate(xs):
        print("Running value {}={} (value {}/{})".format(key, x, i+1, len(xs)))
        kwargs[key] = x
        sub_out_dir = os.path.join(out_dir, str(i+1))
        out_dirs.append(sub_out_dir)
        report = run_single_benchmark(*args, out_dir=sub_out_dir, **kwargs)
        reports[x] = report

    # files
    merge_results = os.path.join(out_dir, 'merge-results.properties')
    joined = os.path.join(out_dir, 'joined.csv')
    reportDir = os.path.join(out_dir, 'report')

    # merge runs together
    inputJtls = [os.path.join(out_dirs[i], 'output.jtl') for i in range(len(xs))]
    prefixes = ["{} {}: ".format(key, x) for x in xs]
    baseJtl = inputJtls[0]
    basePrefix = prefixes[0]
    for i in range(1, len(xs), 3): # MergeResults only joins up to 4 at a time
        with open(merge_results, 'w') as f:
            curInputJtls = [baseJtl] + inputJtls[i:i+3]
            curPrefixes = [basePrefix] + prefixes[i:i+3]
            for j, (jtl, p) in enumerate(zip(curInputJtls, curPrefixes)):
                f.write("inputJtl{}={}\n".format(j+1, jtl))
                f.write("prefixLabel{}={}\n".format(j+1, p))
                f.write("\n")
        merge_call = 'java -jar {} --tool Reporter --generate-csv joined.csv --input-jtl {} --plugin-type MergeResults'.format(CMDRUNNER, merge_results)
        time.sleep(30)
        run_process(merge_call)
        shutil.move('joined.csv', joined) # MergeResults ignores path given and puts result into cwd
        baseJtl = joined
        basePrefix = ""

    # build report
    time.sleep(30)
    run_process('{} -g {} -o {}'.format(JMETER, joined, reportDir))

    print("Merged output available at {}".format(out_dir))
    print("Merged report generated at {}".format(os.path.join(reportDir, 'index.html')))

    return reports


def parseModel():
    if benchmark_model in MODEL_MAP:
        plan, jmeter_args = MODEL_MAP[benchmark_model]
        for k, v in jmeter_args.items():
            if v in RESOURCE_MAP:
                jmeter_args[k] = get_resource(v)
            if k == 'data':
                jmeter_args[k] = os.path.join(TS_BASE, 'benchmarks', v)
        if pargs.input:
            jmeter_args['input_filepath'] = pargs.input[0]
    else:
        plan = JMX_IMAGE_INPUT_MODEL_PLAN
        jmeter_args = {
            'url': benchmark_model,
            'model_name': basename(benchmark_model),
            'input_filepath': pargs.input[0]
        }
    return plan, jmeter_args


def decorate_metrics(data_frame, row_to_read):
    temp_dict = data_frame.loc[row_to_read].to_dict()
    result = dict()
    row_name = row_to_read.replace(' ', '_')
    for key, value in temp_dict.items():
        if key in AGGREGATE_REPORT_CSV_LABELS_MAP:
            new_key = '{}_{}_{}_{}'.format(benchmark_name, benchmark_model, row_name, AGGREGATE_REPORT_CSV_LABELS_MAP[key])
            result[new_key] = value
    return result


class Benchmarks:
    """
    Contains benchmarks to run
    """

    @staticmethod
    def throughput():
        """
        Performs a simple single benchmark that measures the model throughput on inference tasks
        """
        plan, jmeter_args = parseModel()
        return run_single_benchmark(plan, jmeter_args)

    @staticmethod
    def latency():
        """
        Performs a simple single benchmark that measures the model latency on inference tasks
        """
        plan, jmeter_args = parseModel()
        return run_single_benchmark(plan, jmeter_args, threads=1)

    @staticmethod
    def ping():
        """
        Performs a simple ping benchmark that measures the throughput for a ping request to the frontend
        """
        return run_single_benchmark(JMX_PING_PLAN, dict(), threads=5000)

    @staticmethod
    def load():
        """
        Benchmarks number of concurrent inference requests
        """
        plan, jmeter_args = parseModel()
        plan = JMX_CONCURRENT_LOAD_PLAN
        jmeter_args['count'] = 8
        return run_single_benchmark(plan, jmeter_args)

    @staticmethod
    def repeated_scale_calls():
        """
        Benchmarks number of concurrent inference requests
        """
        plan, jmeter_args = parseModel()
        plan = JMX_CONCURRENT_SCALE_CALLS
        jmeter_args['scale_up_workers'] = 16
        jmeter_args['scale_down_workers'] = 2
        return run_single_benchmark(plan, jmeter_args)

    @staticmethod
    def multiple_models():
        """
        Tests with 3 models
        """
        plan = JMX_MULTIPLE_MODELS_LOAD_PLAN
        jmeter_args = {
            'url1': MODEL_MAP[MODEL_RESNET_18][1]['url'],
            'model1_name': MODEL_MAP[MODEL_RESNET_18][1]['model_name'],
            'data3': get_resource('kitten.jpg')
        }
        return run_single_benchmark(plan, jmeter_args)

    @staticmethod
    def concurrent_inference():
        """
        Benchmarks number of concurrent inference requests
        """
        plan, jmeter_args = parseModel()
        return run_multi_benchmark('threads', range(1, 3*5+1, 3), plan, jmeter_args)


def run_benchmark():
    if hasattr(Benchmarks, benchmark_name):
        print("Running benchmark {} with model {}".format(benchmark_name, benchmark_model))
        res = getattr(Benchmarks, benchmark_name)()
        pprint.pprint(res)
        print('\n')
    else:
        raise Exception("No benchmark benchmark_named {}".format(benchmark_name))


def modify_config_props_for_ts(pargs):
    shutil.copyfile(CONFIG_PROP_TEMPLATE, CONFIG_PROP)
    with open(CONFIG_PROP, 'a') as f:
        f.write('\nnumber_of_netty_threads=32')
        f.write('\njob_queue_size=1000')
        if pargs.gpus:
            f.write('\nnumber_of_gpu={}'.format(pargs.gpus[0]))


if __name__ == '__main__':
    benchmark_name_options = [f for f in dir(Benchmarks) if callable(getattr(Benchmarks, f)) and f[0] != '_']
    parser = argparse.ArgumentParser(prog='torchserve-benchmarks', description='Benchmark TorchServe')

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('name', nargs='?', type=str, choices=benchmark_name_options, help='The name of the benchmark to run')
    target.add_argument('-a', '--all', action='store_true', help='Run all benchmarks')
    target.add_argument('-s', '--suite', action='store_true', help='Run throughput and latency on a supplied model')

    model = parser.add_mutually_exclusive_group()
    model.add_argument('-m', '--model', nargs=1, type=str, dest='model', default=[MODEL_RESNET_18], choices=MODEL_MAP.keys(), help='A preloaded model to run.  It defaults to {}'.format(MODEL_RESNET_18))
    model.add_argument('-c', '--custom-model', nargs=1, type=str, dest='model', help='The path to a custom model to run.  The input argument must also be passed. Currently broken')

    parser.add_argument('-d', '--docker', nargs=1, type=str, default=None, help='Docker hub path to use')
    parser.add_argument('-i', '--input', nargs=1, type=str, default=None, help='The input to feed to the test')
    parser.add_argument('-g', '--gpus', nargs=1, type=int, default=None, help='Number of gpus.  Leave empty to run CPU only')

    parser.add_argument('-l', '--loops', nargs=1, type=int, default=[100], help='Number of loops to run')
    parser.add_argument('-t', '--threads', nargs=1, type=int, default=None, help='Number of jmeter threads to run')
    parser.add_argument('-w', '--workers', nargs=1, type=int, default=None, help='Number of TorchServe backend workers to use')

    parser.add_argument('--ts', nargs=1, type=str, help='Target an already running instance of TorchServe instead of spinning up a docker container of TorchServe.  Specify the target with the format address:port (for http) or protocol://address:port')
    parser.add_argument('--management-port', dest='management', nargs=1, type=str, help='When targeting a running TorchServe instance, specify the management port')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display all output')
    parser.add_argument('--options', nargs='*', default=[], help='Additional jmeter arguments.  It should follow the format of --options argname1 argval1 argname2 argval2 ...')
    pargs = parser.parse_args()

    if os.path.exists(OUT_DIR):
        if pargs.all:
            shutil.rmtree(OUT_DIR)
            os.makedirs(OUT_DIR)
    else:
        os.makedirs(OUT_DIR)

    modify_config_props_for_ts(pargs)

    if pargs.suite:
        benchmark_model = pargs.model[0].lower()
        for benchmark_name in BENCHMARK_NAMES:
            run_benchmark()
            if not os.path.isdir(os.path.join(OUT_DIR, benchmark_name, basename(benchmark_model), 'report')):
                run_benchmark()

    elif pargs.all:
        for benchmark_name, benchmark_model in ALL_BENCHMARKS:
            run_benchmark()
    else:
        benchmark_name = pargs.name.lower()
        benchmark_model = pargs.model[0].lower()
        run_benchmark()
