# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Examples for pushing a log to boto client to cloudwatch
"""
import types
import json

import boto3 as boto

from mms.metrics import system_metrics as sys_metric
from mms.metrics.metric_encoder import MetricEncoder


def generate_system_metrics(mod):
    """
    Function acting as a stub for reading a log file, produces similar result
    :param mod:
    :return:
    """
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ != 'collect_all':
            value()

    return json.dumps(sys_metric.system_metrics, indent=4, separators=(',', ':'), cls=MetricEncoder)


def push_cloudwatch(metric_json, client):
    """
    push metric to cloud watch, do some processing.
    :param metric_json:
    :param client:
    :return:
    """
    metrics = json.loads(metric_json)
    cloud_metrics = []
    for metric in metrics:
        cloud_metric = {}
        for key in metric.keys():
            if key != 'RequestId' or key != 'HostName':
                cloud_metric[key] = metric[key]
        cloud_metrics.append(cloud_metric)
    client.put_metric_data(
        Namespace='MXNetModelServer',
        MetricData=cloud_metrics
    )


def connect_cloudwatch():
    client = None
    try:
        client = boto.client('cloudwatch')
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))

    return client


if __name__ == '__main__':
    # Replace this with a log reader
    json_val = generate_system_metrics(sys_metric)
    cloud_client = connect_cloudwatch()
    if cloud_client is not None:
        push_cloudwatch(json_val, cloud_client)
