#!/usr/bin/env bash

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

#Author: Piyush Ghai

set -ex

echo "uploading result files to s3"

hw_type=cpu

if [ "$1" = "True" ]
then
    hw_type=gpu
fi

echo `pwd`
cd /tmp/MMSBenchmark/out
echo `pwd`

today=`date +"%m-%d-%y"`
echo "Saving on S3 bucket on s3://benchmarkai-metrics-prod/daily/mms/$hw_type/$today"

for dir in $(ls `pwd`/)
do
    echo $dir
    aws s3 cp $dir/ s3://benchmarkai-metrics-prod/daily/mms/$hw_type/$today/$dir/ --recursive
done

echo "Files uploaded"
