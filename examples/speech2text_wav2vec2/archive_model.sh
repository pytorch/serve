#!/bin/bash
set -euo pipefail

mkdir -p model_store
# Extra files add all files necessary for processor
torch-model-archiver --model-name Wav2Vec2 --version 1.0 --serialized-file model/pytorch_model.bin --handler ./handler.py --extra-files "model/config.json,model/special_tokens_map.json,model/tokenizer_config.json,model/vocab.json,model/preprocessor_config.json" -f
mv Wav2Vec2.mar model_store