#!/bin/bash
set -euxo pipefail

cd /tmp
rm -rf torchhub.zip
wget https://github.com/nvidia/DeepLearningExamples/archive/torchhub.zip
rm -rf DeepLearningExamples-torchhub
unzip torchhub.zip
cd -
rm -rf tacotron.zip
rm -rf PyTorch
mkdir -p PyTorch/SpeechSynthesis
cp -r /tmp/DeepLearningExamples-torchhub/PyTorch/SpeechSynthesis/* PyTorch/SpeechSynthesis/
zip -r tacotron.zip PyTorch
wget https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/nvidia_tacotron2pyt_fp32_20190427 -O nvidia_tacotron2pyt_fp32_20190427.pth
wget https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427 -O nvidia_waveglowpyt_fp32_20190427.pth
torch-model-archiver --model-name waveglow_synthesizer --version 1.0 --model-file waveglow_model.py --serialized-file nvidia_waveglowpyt_fp32_20190427.pth --handler waveglow_handler.py --extra-files tacotron.zip,nvidia_tacotron2pyt_fp32_20190427.pth -r requirements.txt
rm -rf nvidia_*