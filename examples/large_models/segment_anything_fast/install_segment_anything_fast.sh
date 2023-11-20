#!/bin/bash

# Uninstall torchtext, torchdata, torch, torchvision, and torchaudio
pip uninstall torchtext torchdata torch torchvision torchaudio -y

# Install nightly PyTorch and torchvision from the specified index URL
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed

# Optional: Display the installed PyTorch and torchvision versions
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torchvision; print('torchvision version:', torchvision.__version__)"

echo "PyTorch and torchvision updated successfully!"

# Install the segment-anything-fast package from GitHub
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git

echo "Segment Anything Fast installed successfully!"

echo "Installing other dependencies"
pip install opencv-python matplotlib
sudo apt-get install zlib1g-dev -y
