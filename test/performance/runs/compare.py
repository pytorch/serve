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
Compare artifacts between runs
"""
# pylint: disable=redefined-builtin, self-assigning-variable, broad-except


import csv
import glob
import logging
import sys
import os

import pandas as pd
from junitparser import TestCase, TestSuite, JUnitXml, Skipped, Error, Failure
from runs.taurus import reader as taurus_reader
from runs.storage import LocalStorage, S3Storage

from utils import Timer, get_sub_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


class CompareReportGenerator():

    def __init__(self, path, env_name, local_run):
        self.artifacts_dir = path
        self.current_run_name = os.path.basename(path)
        self.env_name = env_name
        storage_class = LocalStorage if local_run else S3Storage
        self.storage = storage_class(self.artifacts_dir, self.env_name)
        self.junit_reporter = None
        self.pandas_result = None
        self.pass_fail =  True

    def gen(self):
        """Driver method to get comparison directory, do the comparison of it with current run directory
        and then store results
        """
        compare_dir, compare_run_name = self.storage.get_dir_to_compare()
        if compare_run_name:
            self.junit_reporter, self.pandas_result = compare_artifacts(self.storage.artifacts_dir, compare_dir,
                                       self.storage.current_run_name, compare_run_name)
            self.pandas_result.to_csv(os.path.join(self.artifacts_dir, "comparison_result.csv"))
        else:
            logger.warning("The latest run not found for env.")

        self.storage.store_results()
        return self.junit_reporter


class CompareTestSuite():
    """
    Wrapper helper class over JUnit parser Test Suite
    """

    result_types = {"pass": [lambda x: None, "tests"],
                    "fail": [Failure, "failures"],
                    "error": [Error, "errors"],
                    "skip": [Skipped, "skipped"]}

    def __init__(self, name, hostname, t):
        self.ts = TestSuite(name)
        self.ts.errors, self.ts.failures, self.ts.skipped, self.ts.tests = 0, 0, 0, 0
        self.ts.hostname = hostname
        self.ts.timestamp = t.start

    def add_test_case(self, name, msg, type):
        tc = TestCase(name)
        result_type = CompareTestSuite.result_types[type]
        tc.result = result_type[0](msg)
        self.ts.add_testcase(tc)
        setattr(self.ts, result_type[1], getattr(self.ts, result_type[1]) + 1)


def get_log_file(dir, sub_dir):
    """Get metric monitoring log files"""
    metrics_file = os.path.join(dir, sub_dir, "metrics.csv")
    return metrics_file if os.path.exists(metrics_file) else None


def get_aggregate_val(df, agg_func, col):
    """Get aggregate values of a pandas datframe coulmn for given aggregate function"""
    val = None
    if str(col) in df:
        try:
            val = float(getattr(df[str(col)], agg_func)())
        except TypeError:
            val = None
    return val


def compare_values(val1, val2, diff_percent, run_name1, run_name2):
    """ Compare percentage diff values of val1 and val2 """
    if pd.isna(val1) or pd.isna(val2):
        msg = "Either of the value can not be determined. The run1 value is '{}' and " \
              "run2 value is '{}'.".format(val1, val2)
        pass_fail, diff, msg = "error", "NA", msg
    else:
        try:
            if val2 != val1:
                diff = (abs(val2 - val1) / ((val2 + val1) / 2)) * 100

                if diff < float(diff_percent):
                    pass_fail, diff, msg = "pass", diff, "passed"
                else:
                    msg = "The diff_percent criteria has failed. The expected diff_percent is '{}' and actual " \
                          "diff percent is '{}' and the '{}' run value is '{}' and '{}' run value is '{}'. ". \
                        format(diff_percent, diff, run_name1, val1, run_name2, val2)

                    pass_fail, diff, msg = "fail", diff, msg
            else:  # special case of 0
                pass_fail, diff, msg = "pass", 0, ""
        except Exception as e:
            msg = "error while calculating the diff for val1={} and val2={}." \
                  "Error is: {}".format(val1, val2, str(e))
            logger.info(msg)
            pass_fail, diff, msg = "pass", "NA", msg

    return diff, pass_fail, msg


def compare_artifacts(dir1, dir2, run_name1, run_name2):
    """Compare artifacts from dir1 with di2 and store results in out_dir"""

    logger.info("Comparing artifacts from %s with %s", dir1, dir2)
    sub_dirs_1 = get_sub_dirs(dir1)

    over_all_pass = True
    aggregates = ["mean", "max", "min"]
    header = ["run_name1", "run_name2", "test_suite", "metric", "run1", "run2",
              "percentage_diff", "expected_diff", "result", "message"]
    rows = [header]

    reporter = JUnitXml()
    for sub_dir1 in sub_dirs_1:
        with Timer("Comparison test suite {} execution time".format(sub_dir1)) as t:
            comp_ts = CompareTestSuite(sub_dir1, run_name1 + " and " + run_name1, t)

            metrics_file1, metrics_file2 = get_log_file(dir1, sub_dir1), get_log_file(dir2, sub_dir1)
            if not (metrics_file1 and metrics_file2):
                msg = "Metrics monitoring logs are not captured for {} in either " \
                      "of the runs.".format(sub_dir1)
                logger.info(msg)
                rows.append([run_name1, run_name2, sub_dir1, "metrics_log_file_availability",
                             "NA", "NA", "NA", "NA", "pass", msg])
                comp_ts.add_test_case("metrics_log_file_availability", msg, "skip")
                continue

            metrics_from_file1 = pd.read_csv(metrics_file1)
            metrics_from_file2 = pd.read_csv(metrics_file2)
            metrics = taurus_reader.get_compare_metric_list(dir1, sub_dir1)

            for metric_values in metrics:
                col = metric_values[0]
                diff_percent = metric_values[1]
                if diff_percent is None:
                    continue
                for agg_func in aggregates:
                    name = "{}_{}".format(agg_func, str(col))

                    val1 = get_aggregate_val(metrics_from_file1, agg_func, col)
                    val2 = get_aggregate_val(metrics_from_file2, agg_func, col)

                    diff, pass_fail, msg = compare_values(val1, val2, diff_percent, run_name1, run_name2)

                    if over_all_pass:
                        over_all_pass = pass_fail == "pass"

                    result_row = [run_name1, run_name2, sub_dir1, name, val1, val2,
                                  diff, diff_percent, pass_fail, msg]
                    rows.append(result_row)
                    test_name = "{}: diff_percent < {}".format(name, diff_percent)
                    comp_ts.add_test_case(test_name, msg, pass_fail)

            comp_ts.ts.time = t.diff()
            comp_ts.ts.update_statistics()
            reporter.add_testsuite(comp_ts.ts)

    dataframe = pd.DataFrame(rows[1:], columns=rows[0])
    return reporter, dataframe


if __name__ == "__main__":
    compare_artifacts(
    "/Users/mahesh/git/multi-model-server/tests/performance/run_artifacts/xlarge__e436434__1593615831",
    "/Users/mahesh/git/multi-model-server/tests/performance/run_artifacts/xlarge__e436434__1593615748",
    "xlarge__e436434__1593615831", "xlarge__e436434__1593615748"
    )