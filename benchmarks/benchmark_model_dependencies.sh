#!/bin/bash

# Install dependencies and set environment variables for SAM Fast

# Install dependencies
pip install chardet
chmod +x examples/large_models/segment_anything_fast/install_segment_anything_fast.sh
source examples/large_models/segment_anything_fast/install_segment_anything_fast.sh

# Turn off A100G optimization
export SEGMENT_ANYTHING_FAST_USE_FLASH_4=0

echo "Installed dependencies and set environment variables for SAM Fast"

