#!/bin/bash

# This file contains the installation setup for running benchmarks on MAC.
set -ex

echo "Installing JMeter through Brew"
# Script would end on errors, but everything works fine
brew update
brew install jmeter

CELLAR="/usr/local/Cellar/jmeter"

if [ $(ls -1d /usr/local/Cellar/jmeter/* | wc -l) -gt 1 ];then
  echo "Multiple versions of JMeter installed. Exiting..."
  exit 1
fi

JMETER_HOME=`find $CELLAR ! -path $CELLAR -type d -maxdepth 1`

wget https://jmeter-plugins.org/get/ -O $JMETER_HOME/libexec/lib/ext/jmeter-plugins-manager-1.3.jar
wget http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar -O $JMETER_HOME/libexec/lib/cmdrunner-2.2.jar
java -cp $JMETER_HOME/libexec/lib/ext/jmeter-plugins-manager-1.3.jar org.jmeterplugins.repository.PluginManagerCMDInstaller
$JMETER_HOME/libexec/bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
