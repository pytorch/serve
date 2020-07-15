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
File system utilities
"""
# pylint: disable=redefined-builtin, logging-format-interpolation, dangerous-default-value
import logging
import sys
import os
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


def get_sub_dirs(dir, exclude_list=[], include_pattern='*', exclude_pattern=None):
    """Utility method to get list of folders in a directory"""
    dir = dir.strip()
    if not os.path.exists(dir):
        msg = "The path {} does not exit".format(dir)
        logger.error("The path {} does not exit".format(dir))
        raise Exception(msg)

    pattern_list = glob.glob(dir + "/" + include_pattern)
    exclude_pattern_list, exclude_pattern = (glob.glob(dir + "/" + exclude_pattern), exclude_pattern) \
        if exclude_pattern is not None else ([], '')
    skip_pattern = "/skip*"
    skip_list = glob.glob(dir + skip_pattern)

    exclude_patterns = exclude_list
    exclude_patterns.extend([skip_pattern, exclude_pattern])
    logger.info("Excluding the tests with name patterns '{}'.".format("','".join(exclude_patterns)))
    return sorted(list([x for x in os.listdir(dir) if os.path.isdir(dir + "/" + x)
                        and x not in exclude_list
                        and dir + "/" + x in pattern_list
                        and dir + "/" + x not in exclude_pattern_list
                        and dir + "/" + x not in skip_list]))
