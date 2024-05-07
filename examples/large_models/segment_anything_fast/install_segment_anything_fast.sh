#!/bin/bash

# Install the segment-anything-fast package from GitHub
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git

echo "Segment Anything Fast installed successfully!"

echo "Installing other dependencies"
pip install opencv-python matplotlib pycocotools
