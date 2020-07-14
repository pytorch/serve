#!/bin/bash

# Hack needed to make it work with existing benchmark.py
# benchmark.py expects jmeter to be present at a very specific location
mkdir -p /home/linuxbrew/.linuxbrew/Homebrew/Cellar/jmeter/5.3/libexec/
ln -s /opt/apache-jmeter-5.3/lib/ /home/linuxbrew/.linuxbrew/Homebrew/Cellar/jmeter/5.3/libexec/lib
ln -s /opt/apache-jmeter-5.3/bin/ /home/linuxbrew/.linuxbrew/Homebrew/Cellar/jmeter/5.3/libexec/bin

# Create a model store directory
MODEL_STORE_DIR="model_store"
mkdir $MODEL_STORE_DIR

# Start TS and redirect console ouptut and errors to a log file
torchserve --start --model-store=$MODEL_STORE_DIR > ts_console.log 2>&1
sleep 30

# Go to benchmarks directory and trigger suite
cd benchmarks
python benchmark.py latency --ts http://127.0.0.1:8080
EXIT_CODE=$?

torchserve --stop

# Moving TS console log file to logs directory
# Just a convenience for CircleCI to pick up logs from one directory
cd ../
mv ts_console.log logs/

# Exit with the same error code as that of benchmark script
exit $EXIT_CODE