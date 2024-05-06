#!/bin/bash

# Needed for soundfile
sudo apt install libsndfile1 -y

pip install --upgrade transformers sentencepiece datasets[audio] soundfile
