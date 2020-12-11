#!/bin/bash

if [ $# -gt 0 ] 
then
    if [ $1 == "en2fr_model" ] || [ $1 == "en2de_model" ]
    then
        model_name=$1
    else
        echo "Please provide the correct model name while running the script..."
        echo -e "The model names are : \n1. en2fr_model \n2. en2de_model"
        exit 0
    fi
else
    echo "Please provide the model name while running the script..."
    echo -e "The model names are : \n1. en2fr_model \n2. en2de_model"
    exit 0
fi

#installing cython
pip install Cython

#download fairseq repo
#Check if fairseq-build directory does not exist
if [ ! -d "fairseq-build" ] 
then
    git clone https://github.com/pytorch/fairseq
    cd fairseq/
    python setup.py sdist bdist_wheel
    cd ..
    mkdir fairseq-build
    cp fairseq/dist/*.tar.gz fairseq-build/
fi

file_name=$(ls fairseq-build/)

#create requirements.txt file
python requirements_script.py $file_name

#script to create setup_config.json file
python setup_config_script.py $model_name

#Check if a model_store directory does not exist
if [ ! -d "model_store" ] 
then
    mkdir model_store
fi

if [ $model_name == "en2fr_model" ]
then
    #download the En2Fr Translation model checkpoint file
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
    echo
    echo "extracting wmt14.en-fr.joined-dict.transformer.tar.bz2 file ...."
    tar -xvjf wmt14.en-fr.joined-dict.transformer.tar.bz2
    echo "========> extraction completed successfully...."
    echo

    echo
    echo "creating mar file ...."
    torch-model-archiver --model-name TransformerEn2Fr --version 1.0 --serialized-file wmt14.en-fr.joined-dict.transformer/model.pt --export-path model_store --handler model_handler_generalized.py --extra-files wmt14.en-fr.joined-dict.transformer/dict.en.txt,wmt14.en-fr.joined-dict.transformer/dict.fr.txt,wmt14.en-fr.joined-dict.transformer/bpecodes,fairseq-build/$file_name,setup_config.json -r requirements.txt
    echo "========> mar file creation completed successfully...."
    echo

    echo "removing unwanted files ..."
    rm -rf fairseq
    rm wmt14.en-fr.joined-dict.transformer.tar.bz2
    echo "========> removing completed successfully ..."
    echo
else
    #download the En2De Translation model checkpoint file
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz
    echo
    echo "extracting wmt14.en-fr.joined-dict.transformer.tar.bz2 file ...."
    tar -xvf wmt19.en-de.joined-dict.single_model.tar.gz
    echo "========> extraction completed successfully...."
    echo

    echo
    echo "creating mar file ...."
    torch-model-archiver --model-name TransformerEn2De --version 1.0 --serialized-file wmt19.en-de.joined-dict.single_model/model.pt --export-path model_store --handler model_handler_generalized.py --extra-files wmt19.en-de.joined-dict.single_model/dict.en.txt,wmt19.en-de.joined-dict.single_model/dict.de.txt,wmt19.en-de.joined-dict.single_model/bpecodes,fairseq-build/$file_name,setup_config.json -r requirements.txt
    echo "========> mar file creation completed successfully...."
    echo

    echo "removing unwanted files ..."
    rm -rf fairseq
    rm wmt19.en-de.joined-dict.single_model.tar.gz
    echo "========> removing completed successfully ..."
    echo
fi