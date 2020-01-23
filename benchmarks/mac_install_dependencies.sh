#!/bin/bash

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file contains the installation setup for running benchmarks on EC2 isntance.
# To run on a machine with GPU : ./install_dependencies True
# To run on a machine with CPU : ./install_dependencies False
set -ex

echo "Installing JMeter through Brew"
# Script would end on errors, but everything works fine
brew update || {
}
brew install jmeter --with-plugins || {
}

wget https://jmeter-plugins.org/get/ -O /usr/local/Cellar/jmeter/4.0/libexec/lib/ext/jmeter-plugins-manager-1.3.jar
wget http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar -O /usr/local/Cellar/jmeter/4.0/libexec/lib/cmdrunner-2.2.jar
java -cp /usr/local/Cellar/jmeter/4.0/libexec/lib/ext/jmeter-plugins-manager-1.3.jar org.jmeterplugins.repository.PluginManagerCMDInstaller
/usr/local/Cellar/jmeter/4.0/libexec/bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
