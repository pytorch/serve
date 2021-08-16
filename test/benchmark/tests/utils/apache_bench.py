import csv
import os
import time
import re
import boto3
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as n

from inspect import signature
from retrying import retry
from fabric2 import Connection
from botocore.config import Config
from botocore.exceptions import ClientError

from invoke import run, sudo
from invoke.context import Context

from . import DEFAULT_REGION, IAM_INSTANCE_PROFILE, AMI_ID, LOGGER, S3_BUCKET_BENCHMARK_ARTIFACTS

TMP_DIR = "/home/ubuntu"
LOCAL_TMP_DIR = "/tmp"
TS_SERVER_LOG = f"/home/ubuntu/benchmark/logs/model_metrics.log"


class ApacheBenchHandler(object):
    def __init__(self, model_name="benchmark", connection=None):
        self.model_name = model_name
        self.connection = invoke if not connection else connection
        self.local_tmp_dir = os.path.join(LOCAL_TMP_DIR, model_name)
        self.result_file = os.path.join(TMP_DIR, "benchmark/result.txt")
        self.ts_metric_log_file = os.path.join(TMP_DIR, "benchmark/logs/model_metrics.log")
        self.inference_url = "http://127.0.0.1:8080"
        self.install_dependencies()

        self.metrics = {
            "predict.txt": "PredictionTime",
            "handler_time.txt": "HandlerTime",
            "waiting_time.txt": "QueueTime",
            "worker_thread.txt": "WorkerThreadTime",
        }

    @retry(stop_max_attempt_number=7, wait_fixed=60000)
    def install_dependencies(self):
        """
        Installs apache2-utils, assuming it's an Ubuntu instance
        """
        run_out = self.connection.sudo(f"apt install -y apache2-utils", pty=True)
        return run_out.return_code

    def run_apache_bench(self, requests, concurrency, input_file):
        """
        :param requests: number of requests to perform
        :param concurrency: number of virtual users making the request
        :param input_file: assumes the input file needs to be downloaded
        """
        self.connection.run(f"mkdir -p {TMP_DIR}/benchmark")

        if input_file.startswith("https://") or input_file.startswith("http://"):
            self.connection.run(f"wget {input_file}", warn=True)
            file_name = self.connection.run(f"basename {input_file}").stdout.strip()
            # Copy to the directory with other benchmark artifacts
            self.connection.run(f"cp {file_name} {os.path.join(TMP_DIR, 'benchmark/input')}")
        else:
            self.connection.run(f"cp {input_file} {os.path.join(TMP_DIR, 'benchmark/input')}")

        # Run warmup
        apache_bench_warmup_command = f"ab -c {concurrency} {100} -k -p {TMP_DIR}/benchmark/input -T application/jpg {self.inference_url}/predictions/benchmark"
        run_out = self.connection.run(apache_bench_warmup_command, warn=True, pty=True)
        LOGGER.info(f"warmup command used: {apache_bench_warmup_command}")

        apache_bench_command = f"ab -c {concurrency} -n {requests} -k -p {TMP_DIR}/benchmark/input -T application/jpg {self.inference_url}/predictions/benchmark > {self.result_file}"

        # Run apache bench
        run_out = self.connection.run(apache_bench_command, warn=True, pty=True)

        LOGGER.info(f"apache bench command used: {apache_bench_command}")

        time.sleep(40)

        LOGGER.error(f"{run_out.stdout}")
        if run_out.return_code != 0:
            LOGGER.error(f"apache bench command failed.")

    def clean_up(self):
        self.connection.run(f"rm -rf {os.path.join(TMP_DIR, 'benchmark')}")

    def extract_metrics(self, connection=None):
        metric_log = f"{os.path.join(self.local_tmp_dir, 'benchmark_metric.log')}"
        result_file = f"{os.path.join(self.local_tmp_dir, 'result.txt')}"

        temp_uuid = uuid.uuid4()

        time.sleep(5)
        
        # Upload to s3 and fetch back to local instance: more reliable than using self.connection.get()
        connection.run(f"aws s3 cp {self.result_file} {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/result.txt")
        time.sleep(2)
        run(f"aws s3 cp {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/result.txt {result_file}")

        time.sleep(2)
        connection.run(f"aws s3 cp {TS_SERVER_LOG} {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/benchmark_metric.log")
        time.sleep(2)
        run(f"aws s3 cp {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/benchmark_metric.log {metric_log}")

        # Clean up right away
        run(f"aws s3 rm --recursive {S3_BUCKET_BENCHMARK_ARTIFACTS}/{temp_uuid}/")

        with open(metric_log) as f:
            lines = f.readlines()

        for k, v in self.metrics.items():
            all_lines = []
            pattern = re.compile(v)
            for line in lines:
                if pattern.search(line):
                    all_lines.append(line.split("|")[0].split(":")[3].strip())

            out_fname = f"{self.local_tmp_dir}/{k}"
            LOGGER.info(f"\nWriting extracted {v} metrics to {out_fname} ")
            with open(out_fname, "w") as outf:
                all_lines = map(lambda x: x + "\n", all_lines)
                outf.writelines(all_lines)

    def extract_entity(self, data, pattern, index, delim=" "):
        pattern = re.compile(pattern)
        for line in data:
            if pattern.search(line):
                return line.split(delim)[index].strip()

    def generate_csv_output(self, requests, concurrency, connection=None):
        LOGGER.info("*Generating CSV output...")

        batched_requests = requests / concurrency
        line50 = int(batched_requests / 2)
        line90 = int(batched_requests * 9 / 10)
        line99 = int(batched_requests * 99 / 100)
        artifacts = {}
        with open(f"{self.local_tmp_dir}/result.txt") as f:
            data = f.readlines()
        artifacts["Benchmark"] = "AB"
        artifacts["Model"] = self.model_name
        artifacts["Concurrency"] = concurrency
        artifacts["Requests"] = requests
        artifacts["TS failed requests"] = self.extract_entity(data, "Failed requests:", -1)
        artifacts["TS throughput"] = self.extract_entity(data, "Requests per second:", -3)
        artifacts["TS latency P50"] = self.extract_entity(data, "50%", -1)
        artifacts["TS latency P90"] = self.extract_entity(data, "90%", -1)
        artifacts["TS latency P99"] = self.extract_entity(data, "99%", -1)
        artifacts["TS latency mean"] = self.extract_entity(data, "Time per request:.*mean\)", -3)
        artifacts["TS error rate"] = int(artifacts["TS failed requests"]) / int(requests) * 100

        with open(os.path.join(self.local_tmp_dir, "predict.txt")) as f:
            lines = f.readlines()
            lines.sort(key=float)
            artifacts["Model_p50"] = lines[line50].strip()
            artifacts["Model_p90"] = lines[line90].strip()
            artifacts["Model_p99"] = lines[line99].strip()

        for m in self.metrics:
            df = pd.read_csv(f"{self.local_tmp_dir}/{m}", header=None, names=["data"])
            artifacts[m.split(".txt")[0] + "_mean"] = df["data"].values.mean().round(2)

        with open(os.path.join(self.local_tmp_dir, "ab_report.csv"), "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(artifacts.keys())
            csvwriter.writerow(artifacts.values())

        LOGGER.info(f"Generated csv output.")

        return artifacts

    def generate_report(self, requests, concurrency, connection=None):
        self.extract_metrics(connection=connection)
        self.generate_csv_output(requests, concurrency, connection=connection)
        # self.generate_latency_graph()
        # self.generate_profile_graph()
        pass
