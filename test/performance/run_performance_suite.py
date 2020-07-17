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
Run Performance Regression Test Cases and Generate Reports
"""
# pylint: disable=redefined-builtin, no-value-for-parameter, unused-argument

import logging
import os
import subprocess
import sys
import time
import pathlib

import click
from tqdm import tqdm

from runs.context import ExecutionEnv
from runs.taurus import get_taurus_options, x2junit, update_taurus_metric_files
from utils import run_process, Timer, get_sub_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

ROOT_PATH = pathlib.Path(__file__).parent.absolute()
RUN_ARTIFACTS_PATH = os.path.join(ROOT_PATH, "run_artifacts")
GLOBAL_CONFIG_PATH = os.path.join(ROOT_PATH, "tests", "global_config.yaml")
MONITORING_AGENT = os.path.join(ROOT_PATH, "agents", "metrics_monitoring_server.py")


def get_artifacts_dir(ctx, param, value):
    commit_id = subprocess.check_output('git rev-parse --short HEAD'.split()).decode("utf-8")[:-1]
    run_name = "{}__{}__{}".format(ctx.params['env_name'], commit_id, int(time.time()))
    if value is None:
        artifacts_dir = os.path.join(RUN_ARTIFACTS_PATH, run_name)
    else:
        artifacts_dir = os.path.abspath(value)
        artifacts_dir = os.path.join(artifacts_dir, run_name)
    return artifacts_dir


def validate_env(ctx, param, value):
    try:
        if '__' in value:
            raise ValueError
        return value
    except ValueError:
        raise click.BadParameter('Environment name should not have double underscores in it.')


@click.command()
@click.option('-a', '--artifacts-dir', help='Directory to store artifacts.', type=click.Path(writable=True),
              callback=get_artifacts_dir)
@click.option('-t', '--test-dir', help='Directory containing tests.', type=click.Path(exists=True),
              default=os.path.join(ROOT_PATH, "tests"))
@click.option('-p', '--pattern', help='Test case folder name glob pattern', default="*")
@click.option('-x', '--exclude-pattern', help='Test case folder name glob pattern to exclude', default=None)
@click.option('-j', '--jmeter-path', help='JMeter executable path.')
@click.option('-e', '--env-name', help='Environment filename without the extension. Contains threshold values.',
              required=True, callback=validate_env)
@click.option('--monit/--no-monit', help='Start Monitoring server', default=True)
@click.option('--compare-local/--no-compare-local', help='Compare with previous run with files stored'
                                                         ' in artifacts directory', default=True)
@click.option('-c', '--compare-with', help='Compare with commit id, branch, tag, HEAD~N.', default="HEAD~1")
def run_test_suite(artifacts_dir, test_dir, pattern, exclude_pattern,
                   jmeter_path, env_name, monit, compare_local, compare_with):
    """Collect test suites, run them and generate reports"""

    logger.info("Artifacts will be stored in directory %s", artifacts_dir)
    test_dirs = get_sub_dirs(test_dir, exclude_list=[], include_pattern=pattern,
                             exclude_pattern=exclude_pattern)
    if not test_dirs:
        logger.info("No test cases are collected...Exiting.")
        sys.exit(3)
    else:
        logger.info("Collected tests %s", test_dirs)

    with ExecutionEnv(MONITORING_AGENT, artifacts_dir, env_name, compare_local, compare_with, monit) as prt:
        pre_command = 'export PYTHONPATH={}:$PYTHONPATH;'.format(os.path.join(str(ROOT_PATH), "runs", "taurus", "override"))
        for suite_name in tqdm(test_dirs, desc="Test Suites"):
            with Timer("Test suite {} execution time".format(suite_name)) as t:
                suite_artifacts_dir = os.path.join(artifacts_dir, suite_name)
                options_str = get_taurus_options(suite_artifacts_dir, jmeter_path)
                env_yaml_path = os.path.join(test_dir, suite_name, "environments", "{}.yaml".format(env_name))
                env_yaml_path = "" if not os.path.exists(env_yaml_path) else env_yaml_path
                test_file = os.path.join(test_dir, suite_name, "{}.yaml".format(suite_name))
                with x2junit.X2Junit(suite_name, suite_artifacts_dir, prt.reporter, t, env_name) as s:
                    s.code, s.err = run_process("{} bzt {} {} {} {}".format(pre_command, options_str,
                                                                            GLOBAL_CONFIG_PATH, test_file,
                                                                            env_yaml_path))

                    update_taurus_metric_files(suite_artifacts_dir, test_file)

    sys.exit(prt.exit_code)


if __name__ == "__main__":
    run_test_suite()
