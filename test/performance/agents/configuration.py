#!/usr/bin/env python3

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
Read configuration file
"""
# pylint: disable=redefined-builtin, bare-except
import os
import configparser
import pathlib

config = configparser.ConfigParser()
path = pathlib.Path(__file__).parent.absolute()
config.read(os.path.join(path, 'config.ini'))


def get(section, key, default=''):
    try:
        return config[section][key]
    except:
        return default
