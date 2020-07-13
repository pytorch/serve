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
Convert the Taurus Test suite XML to Junit XML
"""
# pylint: disable=redefined-builtin


import os
import pandas as pd
from runs.taurus.reader import get_compare_metric_list

import html
import tabulate
from bzt.modules.passfail import DataCriterion
from junitparser import TestCase, TestSuite, JUnitXml, Skipped, Error, Failure, Pass


class X2Junit(object):
    """
       Context Manager class to do convert Taurus Test suite XML report which is in Xunit specifications
       to JUnit XML report.
    """
    def __init__(self, name, artifacts_dir, junit_xml, timer, env_name):
        self.ts = TestSuite(name)
        self.name = name
        self.junit_xml = junit_xml
        self.timer = timer
        self.artifacts_dir = artifacts_dir
        self.env_name = env_name
        self.metrics = None
        self.metrics_agg_dict = {}

        self.code = 0
        self.err = ""

        self.ts.tests, self.ts.failures, self.ts.skipped, self.ts.errors = 0, 0, 0, 0

    def __enter__(self):
        return self

    def add_compare_tests(self):
        compare_list = get_compare_metric_list(self.artifacts_dir, "")
        for metric_values in compare_list:
            col = metric_values[0]
            diff_percent = metric_values[2]
            tc = TestCase("{}_diff_run > {}".format(col, diff_percent))
            if diff_percent is None:
                tc.result = Skipped("diff_percent_run value is not mentioned")
                self.ts.skipped += 1
            elif self.metrics is None:
                tc.result = Skipped("Metrics are not captured")
                self.ts.skipped += 1
            else:
                col_metric_values = getattr(self.metrics, col, None)
                if col_metric_values is None:
                    tc.result = Error("Metric {} is not captured".format(col))
                    self.ts.errors += 1
                elif len(col_metric_values) < 2:
                    tc.result = Skipped("Enough values are not captured")
                    self.ts.errors += 1
                else:
                    first_value = col_metric_values.iloc[0]
                    last_value = col_metric_values.iloc[-1]

                    try:
                        if last_value == first_value == 0:
                            diff_actual = 0
                        else:
                            diff_actual = (abs(last_value - first_value) / ((last_value + first_value) / 2)) * 100

                        if float(diff_actual) <= float(diff_percent):
                            self.ts.tests += 1
                        else:
                            tc.result = Failure("The first value and last value of run are {}, {} "
                                                "with precent diff {}".format(first_value, last_value, diff_actual))

                    except Exception as e:
                        tc.result = Error("Error while comparing values {}".format(str(e)))
                        self.ts.errors += 1

            self.ts.add_testcase(tc)

    @staticmethod
    def casename_to_criteria(test_name):
        metric = None
        if ' of ' not in test_name:
            test_name = "label of {}".format(test_name)
        try:
            test_name = html.unescape(html.unescape(test_name))
            criteria = DataCriterion.string_to_config(test_name)
        except Exception as e:
            return None

        label = criteria["label"].split('/')
        if len(label) == 2:
            metric = label[1]
        return metric

    def percentile_values(self, metric_name):
        values = {}
        if self.metrics is not None and metric_name is not None:
            metric_vals = getattr(self.metrics, metric_name, None)
            if metric_vals is not None:
                centile_values = [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1]
                for centile in centile_values:
                    val = getattr(metric_vals, 'quantile')(centile)
                    values.update({str(centile * 100)+"%": val})

        return values

    def update_metrics(self):
        metrics_file = os.path.join(self.artifacts_dir, "metrics.csv")
        rows = []
        agg_dict = {}
        if os.path.exists(metrics_file):
            self.metrics = pd.read_csv(metrics_file)
            centile_values = [0, 0.5, 0.9, 0.95, 0.99, 0.999, 1]
            header_names = [str(colname) + "%" for colname in centile_values]
            if self.metrics.size:
                 for col in self.metrics.columns:
                     row = [str(col)]
                     metric_vals = getattr(self.metrics, str(col), None)
                     for centile in centile_values:
                         row.append(getattr(metric_vals, 'quantile')(centile))
                     agg_dict.update({row[0]: dict(zip(header_names, row[1:]))})
                     rows.append(row)

                 header = ["metric_name"]
                 header.extend(header_names)
                 dataframe = pd.DataFrame(rows, columns=header)
                 print("Metric percentile values:\n")
                 print(tabulate.tabulate(rows, headers=header, tablefmt="grid"))
                 dataframe.to_csv(os.path.join(self.artifacts_dir, "metrics_agg.csv"))

        self.metrics_agg_dict = agg_dict

    def __exit__(self, type, value, traceback):
        print("error code is "+str(self.code))

        self.update_metrics()
        xunit_file = os.path.join(self.artifacts_dir, "xunit.xml")
        if self.code == 1:
            tc = TestCase(self.name)
            tc.result = Error(self.err)
            self.ts.add_testcase(tc)
        elif os.path.exists(xunit_file):
            xml = JUnitXml.fromfile(xunit_file)
            for i, suite in enumerate(xml):
                for case in suite:
                    name = "scenario_{}: {}".format(i, case.name)
                    result = case.result

                    metric_name = X2Junit.casename_to_criteria(case.name)
                    values = self.metrics_agg_dict.get(metric_name, None)
                    msg = result.message if result else ""
                    if values:
                        val_msg = "Actual percentile values are {}".format(values)
                        msg = "{}. {}".format(msg, val_msg)

                    if isinstance(result, Error):
                        self.ts.failures += 1
                        result = Failure(msg, result.type)
                    elif isinstance(result, Failure):
                        self.ts.errors += 1
                        result = Error(msg, result.type)
                    elif isinstance(result, Skipped):
                        self.ts.skipped += 1
                        result = Skipped(msg, result.type)
                    else:
                        self.ts.tests += 1

                    tc = TestCase(name)
                    tc.result = result
                    self.ts.add_testcase(tc)
        else:
            tc = TestCase(self.name)
            tc.result = Skipped("Skipped criteria test cases as Taurus XUnit file is not generated.")
            self.ts.add_testcase(tc)

        self.add_compare_tests()

        self.ts.hostname = self.env_name
        self.ts.timestamp = self.timer.start
        self.ts.time = self.timer.diff()
        self.ts.update_statistics()
        self.junit_xml.add_testsuite(self.ts)

        # Return False needed so that __exit__ method do no ignore the exception
        # otherwise exception are not reported
        return False

if __name__ == "__main__":
    from utils.timer import Timer
    with Timer("ads") as t:
        test_folder = '/Users/demo/git/serve/test/performance/'\
                        'run_artifacts/xlarge__2dc700f__1594662587/scale_down_workers'
        x = X2Junit("test", test_folder, JUnitXml(), t, "xlarge")

    # x.update_metrics()
    #
    # x.add_compare_tests()

    x.__exit__(None, None, None)
    x.ts

    print("a")


