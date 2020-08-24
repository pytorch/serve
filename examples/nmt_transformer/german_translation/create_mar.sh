#!/bin/bash

rm -rf fairseq
rm wmt19.en-de.joined-dict.single_model.tar.gz
rm -rf fairseq-build

#installing cython
pip install Cython

#download fairseq repo
git clone https://github.com/pytorch/fairseq
cd fairseq/
python3 setup.py sdist bdist_wheel
cd ..
mkdir fairseq-build
cp fairseq/dist/*.tar.gz fairseq-build/
file_name=$(ls fairseq-build/)

#create requirements.txt file
python3 requirement_script.py $file_name

#download the En2Fr Translation model checkpoint file
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz
echo
echo "extracting wmt14.en-fr.joined-dict.transformer.tar.bz2 file ...."
tar -xvf wmt19.en-de.joined-dict.single_model.tar.gz
echo "========> extraction completed successfully...."
echo

echo
echo "creating mar file ...."
mkdir model_store
torch-model-archiver --model-name TransformerEn2De --version 1.0 --serialized-file wmt19.en-de.joined-dict.single_model/model.pt --export-path model_store --handler model_handler.py --extra-files wmt19.en-de.joined-dict.single_model/dict.en.txt,wmt19.en-de.joined-dict.single_model/dict.de.txt,wmt19.en-de.joined-dict.single_model/bpecodes,fairseq-build/$file_name -r requirements.txt
echo "========> mar file creation completed successfully...."
echo

echo "removing unwanted files ..."
rm -rf fairseq
rm wmt19.en-de.joined-dict.single_model.tar.gz
echo "========> removing completed successfully ..."
echo
