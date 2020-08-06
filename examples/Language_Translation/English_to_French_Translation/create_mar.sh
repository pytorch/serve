#!/bin/bash
set -euxo pipefail

rm -rf fairseq
rm wmt14.en-fr.joined-dict.transformer.tar.bz2

git clone https://github.com/pytorch/fairseq.git
# cd fairseq/
# pip install ./

# cd ..
echo
echo "extracting wmt14.en-fr.joined-dict.transformer.tar.bz2 file ...."
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
tar -xvjf wmt14.en-fr.joined-dict.transformer.tar.bz2
echo "========> extraction completed successfully...."
echo

echo
echo "creating mar file ...."
mkdir model_store
torch-model-archiver --model-name TransformerEn2Fr --version 1.0 --serialized-file wmt14.en-fr.joined-dict.transformer/model.pt --export-path model_store --handler model_handler.py
echo "========> mar file creation completed successfully...."
echo

echo "removing unwanted files ..."
# rm -rf fairseq
rm wmt14.en-fr.joined-dict.transformer.tar.bz2
echo "========> removing completed successfully ..."