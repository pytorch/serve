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


class CloudWatchMetricsHandler:
    def __init__(self, context="DevTest", sub_namespace="TestModel"):
        self.client = boto3.Session(region_name=DEFAULT_REGION).client("cloudwatch")
        self.context = context
        self.namespace = f"TorchServe/{context.title()}/{sub_namespace.title()}"

    def push(self, name, unit, value, metrics_info):
        # dimensions = [{"Name": "BenchmarkContextTest", "Value":self.context}]
        dimensions = []

        for key in metrics_info:
            dimensions.append({"Name": key, "Value": str(metrics_info.get(key))})

        try:
            response = self.client.put_metric_data(
                MetricData=[{"MetricName": name, "Dimensions": dimensions, "Unit": unit, "Value": float(value)}],
                Namespace=self.namespace,
            )
        except Exception as e:
            raise Exception(str(e))

        return response

    def push_benchmark_metrics(self, benchmark_dict):

        # CloudWatch allows a maximum of 10 dimensions for a metric, so only the most important are published here
        info = {"Instance Type": benchmark_dict.get("instance_type"), "Batch Size": benchmark_dict.get("Batch Size")}

        self.push("Model Latency P90", "Milliseconds", benchmark_dict.get("Model_p90"), info)
        self.push("TS Throughput", "Count/Second", benchmark_dict.get("TS throughput"), info)
        self.push("TS Error Rate", "Count/Second", str(benchmark_dict.get("TS error rate")), info)

        LOGGER.info(f"Benchmark metric pushed to cloudwatch")
