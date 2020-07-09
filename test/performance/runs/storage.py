#!/usr/bin/env python

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Result store classes
"""
# pylint: disable=redefined-builtin


import logging
import os
import sys
import shutil

import boto3
import pathlib
from agents import configuration

from utils import run_process

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
S3_BUCKET = configuration.get('suite', 's3_bucket')


class Storage():
    """Class to store and retrieve artifacts"""

    def __init__(self, path, env_name):
        self.artifacts_dir = path
        self.current_run_name = os.path.basename(path)
        self.env_name = env_name

    def get_dir_to_compare(self):
        """get the artifacts dir to compare to"""

    def store_results(self):
        """Store the results"""

    @staticmethod
    def get_latest(names, env_name, exclude_name):
        """
        Get latest directory for same env_name name given a list of them.
        :param names: list of folder names in the format env_name___commitid__timestamp
        :param env_name: filter for env_name
        :param exclude_name: any name to exclude
        :return: latest directory name
        """
        max_ts = 0
        latest_run = ''
        for run_name in names:
            run_name_list = run_name.split('__')
            if env_name == run_name_list[0] and run_name != exclude_name:
                if int(run_name_list[2]) > max_ts:
                    max_ts = int(run_name_list[2])
                    latest_run = run_name

        return latest_run


class LocalStorage(Storage):
    """
    Compare the monitoring metrics for current and previous run for the same env_name
    """

    def get_dir_to_compare(self):
        """Get latest run directory name to be compared with"""
        parent_dir = pathlib.Path(self.artifacts_dir).parent
        names = [di for di in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, di))]
        latest_run = self.get_latest(names, self.env_name, self.current_run_name)
        return os.path.join(parent_dir, latest_run), latest_run


class S3Storage(Storage):
    """Compare current run results with the results stored on S3"""

    def get_dir_to_compare(self):
        """Get latest run result artifacts directory  for same env_name from S3 bucket
        and store it locally for further comparison
        """
        comp_data_path = os.path.join(self.artifacts_dir, "comp_data")
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(S3_BUCKET)
        result = bucket.meta.client.list_objects(Bucket=bucket.name,
                                                 Delimiter='/')
        run_names = []
        for o in result.get('CommonPrefixes'):
            run_names.append(o.get('Prefix')[:-1])

        latest_run = self.get_latest(run_names, self.env_name, self.current_run_name)
        if not latest_run:
            logger.info("No run found for env_id %s", self.env_name)
            return '', ''

        if not os.path.exists(comp_data_path):
            os.makedirs(comp_data_path)

        tgt_path = os.path.join(comp_data_path, latest_run)
        run_process("aws s3 cp  s3://{}/{} {} --recursive".format(bucket.name, latest_run, tgt_path))

        return tgt_path, latest_run

    def store_results(self):
        """Store the run results back to S3"""
        comp_data_path = os.path.join(self.artifacts_dir, "comp_data")
        if os.path.exists(comp_data_path):
            shutil.rmtree(comp_data_path)

        run_process("aws s3 cp {} s3://{}/{}  --recursive".format(self.artifacts_dir, S3_BUCKET,
                                                                  self.current_run_name))
