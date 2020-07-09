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

import os
import html
import textwrap
import tabulate
from utils import run_process
from junitparser import JUnitXml

header = ["suite_name", "test_case", "result", "message"]


class JunitConverter():

    def __init__(self, junit_xml, out_dir, report_name):
        self.junit_xml = junit_xml
        self.junit_xml_path = os.path.join(out_dir, '{}.xml'.format(report_name))
        self.junit_html_path = os.path.join(out_dir, '{}.html'.format(report_name))

    def generate_junit_report(self):
        self.junit_xml.update_statistics()
        self.junit_xml.write(self.junit_xml_path)
        # vjunit pip package is used here
        run_process("vjunit -f {} -o {}".format(self.junit_xml_path, self.junit_html_path))


def pretty_text(data):
    """Unsescape the html characters from the data & wrap it"""
    if data is not None:
        return textwrap.fill(html.unescape(html.unescape(data)), width=60)
    else:
        return ""


def junit2array(junit_xml):
    """convert junit xml junitparser.JUnitXml object to 2d array """
    rows = [header]
    for i, suite in enumerate(junit_xml):
        if len(suite) == 0:
            rows.append([suite.name, "", "skipped",
                         "No criteria specified or there is an error."])
        else:
            for case in suite:
                result = case.result
                tag, msg = (result._tag, result.message) if result else ("passed", "")
                rows.append([suite.name, pretty_text(case.name), tag, pretty_text(msg)])

    return rows


def junit2tabulate(junit_xml):
    """convert junit xml junitparser.JUnitXml object or a Junit xml to tabulate string """
    if not isinstance(junit_xml, JUnitXml):
        if os.path.exists(junit_xml):
            junit_xml = JUnitXml.fromfile(junit_xml)
        else:
            return tabulate.tabulate([[header]], headers='firstrow')
    data = junit2array(junit_xml)
    return tabulate.tabulate(data, headers='firstrow', tablefmt="grid")
