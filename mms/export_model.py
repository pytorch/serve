# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This command line interface is no longer used. Please refer to model-archiver tool for the new CLI for exporting models.
"""


def main():
    print('\033[93m'  # Red Color start
          + "mxnet-model-export is no longer supported.\n"
            "Please use model-archiver to create 1.0 model archive.\n"
            "For more detail, see: https://pypi.org/project/model-archiver"
          + '\033[0m')  # Red Color end


if __name__ == '__main__':
    main()
