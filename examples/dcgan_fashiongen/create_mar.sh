#!/bin/bash

REPONAME="pytorch_GAN_zoo"
SHA="b75dee40918caabb4fe7ec561522717bf096a8cb" #master branch as of 27th Aug 2020
SRCZIP="$SHA.zip"
SRCDIR="$REPONAME-$SHA"
MODELSZIP="models.zip"
MODELSDIR="models"
CHECKPOINT="DCGAN_fashionGen-1d67302.pth" #The DCGAN pretrained model as of 27th Aug 2020
CHECKPOINT_RENAMED="DCGAN_fashionGen.pth"

# Clean Up before exit
function cleanup {
  rm -rf $SRCZIP $SRCDIR $MODELSZIP $CHECKPOINT_RENAMED $MODELSDIR
}
trap cleanup EXIT

# Download and Extract model's source code
sudo apt-get install zip unzip -y

wget https://github.com/facebookresearch/pytorch_GAN_zoo/archive/$SRCZIP
unzip $SRCZIP
# Get the models directory from the source code and zip it up
# This will later be used by torchserve for loading the model
mv $SRCDIR/models .
zip -r $MODELSZIP $MODELSDIR

# Download checkpoint
wget https://dl.fbaipublicfiles.com/gan_zoo/$CHECKPOINT -O $CHECKPOINT_RENAMED

# Create *.mar
torch-model-archiver --model-name dcgan_fashiongen \
                     --version 1.0 \
                     --serialized-file $CHECKPOINT_RENAMED \
                     --handler dcgan_fashiongen_handler.py \
                     --extra-files $MODELSZIP \
                     --force
