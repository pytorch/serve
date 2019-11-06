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
Single start point for system metrics and process metrics script

"""
import logging
import sys

from mms.metrics import system_metrics
from mms.metrics.process_memory_metric import check_process_mem_usage

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    system_metrics.collect_all(sys.modules['mms.metrics.system_metrics'])

    check_process_mem_usage(sys.stdin)
