#!/bin/bash

# This file contains the installation setup for running benchmarks on MAC.
set -ex

echo "Installing JMeter through Brew"
# Script would end on errors, but everything works fine
brew update
brew install jmeter

wget https://jmeter-plugins.org/get/ -O /usr/local/Cellar/jmeter/5.2.1/libexec/lib/ext/jmeter-plugins-manager-1.3.jar
wget http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar -O /usr/local/Cellar/jmeter/5.2.1/libexec/lib/cmdrunner-2.2.jar
java -cp /usr/local/Cellar/jmeter/5.2.1/libexec/lib/ext/jmeter-plugins-manager-1.3.jar org.jmeterplugins.repository.PluginManagerCMDInstaller
/usr/local/Cellar/jmeter/5.2.1/libexec/bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
