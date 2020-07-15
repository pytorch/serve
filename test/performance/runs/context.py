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
Start and stop monitoring server
"""
# pylint: disable=redefined-builtin

import logging
import os
import sys
import time
import subprocess
import webbrowser
from termcolor import colored

from junitparser import JUnitXml
from runs.compare import CompareReportGenerator
from runs.junit import JunitConverter, junit2tabulate

from utils import run_process

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


def get_git_commit_id(compare_with):
    """Get short commit id for compare_with commit, branch, tag"""
    cmd = 'git rev-parse --short {}'.format(compare_with)
    logger.info("Running command: %s", cmd)
    commit_id = subprocess.check_output(cmd.split()).decode("utf-8")[:-1]
    logger.info("Commit id for compare_with='%s' is '%s'", compare_with, commit_id)
    return commit_id


class ExecutionEnv(object):
    """
    Context Manager class to run the performance regression suites
    """

    def __init__(self, agent, artifacts_dir, env, local_run, compare_with, use=True, check_model_server_status=False):
        self.monitoring_agent = agent
        self.artifacts_dir = artifacts_dir
        self.use = use
        self.env = env
        self.local_run = local_run
        self.compare_with = get_git_commit_id(compare_with)
        self.check_model_server_status = check_model_server_status
        self.reporter = JUnitXml()
        self.compare_reporter_generator = CompareReportGenerator(self.artifacts_dir, self.env, self.local_run,
                                                                 compare_with)
        self.exit_code = 1

    def __enter__(self):
        if self.use:
            start_monitoring_server = "{} {} --start".format(sys.executable, self.monitoring_agent)
            run_process(start_monitoring_server, wait=False)
            time.sleep(2)
        return self

    @staticmethod
    def open_report(file_path):
        """Open html report in browser """
        if os.path.exists(file_path):
            return webbrowser.open_new_tab('file://' + os.path.realpath(file_path))
        return False

    @staticmethod
    def report_summary(reporter, suite_name):
        """Create a report summary """
        if reporter and os.path.exists(reporter.junit_html_path):
            status = reporter.junit_xml.errors or reporter.junit_xml.failures
            status, code, color = ("failed", 3, "red") if status else ("passed", 0, "green")

            msg = "{} run has {}.".format(suite_name, status)
            logger.info(colored(msg, color, attrs=['reverse', 'blink']))
            logger.info("%s report - %s", suite_name, reporter.junit_html_path)
            logger.info("%s summary:", suite_name)
            print(junit2tabulate(reporter.junit_xml))
            ExecutionEnv.open_report(reporter.junit_html_path)
            return code

        else:
            msg = "{} run report is not generated.".format(suite_name)
            logger.info(colored(msg, "yellow", attrs=['reverse', 'blink']))
            return 0

    def __exit__(self, type, value, traceback):
        if self.use:
            stop_monitoring_server = "{} {} --stop".format(sys.executable, self.monitoring_agent)
            run_process(stop_monitoring_server)

        junit_reporter = JunitConverter(self.reporter, self.artifacts_dir, 'performance_results')
        junit_reporter.generate_junit_report()
        junit_compare = self.compare_reporter_generator.gen()
        junit_compare_reporter = None
        if junit_compare:
            junit_compare_reporter = JunitConverter(junit_compare, self.artifacts_dir, 'comparison_results')
            junit_compare_reporter.generate_junit_report()

        compare_exit_code = ExecutionEnv.report_summary(junit_compare_reporter, "Comparison Test suite")
        exit_code = ExecutionEnv.report_summary(junit_reporter, "Performance Regression Test suite")

        self.exit_code = 0 if 0 == exit_code == compare_exit_code else 3

        # Return True needed so that __exit__ method do no ignore the exception
        # otherwise exception are not reported
        return False
