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
    def __init__(self, context="dashboard", namespace="TORCHSERVE_DEV_METRICS"):
        self.client = boto3.Session(region_name=DEFAULT_REGION).client("cloudwatch")
        self.context = context
        self.namespace = f"TORCHSERVE_{context.upper()}_METRICS_TEST"

    def push(self, name, unit, value, metrics_info):
        # dimensions = [{"Name": "BenchmarkContextTest", "Value":self.context}]
        dimensions = []

        for key in metrics_info:
            dimensions.append({"Name": key, "Value": metrics_info.get(key)})

        try:
            response = self.client.put_metric_data(
                MetricData=[{"MetricName": name, "Dimensions": dimensions, "Unit": unit, "Value": float(value)}],
                Namespace=self.namespace,
            )
        except Exception as e:
            raise Exception(str(e))

        return response

    def push_benchmark_metrics(self, metric_name, unit, benchmark_dict):

        # Extract value and dimensions from benchmark_dict

        value = benchmark_dict.get("Model_p90")

        # CloudWatch allows a maximum of 10 dimensions for a metric, so only the most important are published here
        info = {
            "Model_latency_p50": benchmark_dict.get("Model_p50"),
            "Model_latency_p90": benchmark_dict.get("Model_p90"),
            "Model_latency_p99": benchmark_dict.get("Model_p99"),
            "TS_latency_p50": benchmark_dict.get("TS latency P50"),
            "TS_latency_p90": benchmark_dict.get("TS latency P90"),
            "TS_latency_p99": benchmark_dict.get("TS latency P99"),
            "TS_throughput": benchmark_dict.get("TS throughput"),
            "TS_error_rate": str(benchmark_dict.get("TS error rate")),
        }

        self.push(metric_name, unit, value, info)

        LOGGER.info(f"Benchmark metrics for {metric_name} pushed to cloudwatch")
