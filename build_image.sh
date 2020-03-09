#!/bin/bash

MACHINE=cpu
BRANCH_NAME="master"

for arg in "$@"
do
    case $arg in
        -h|--help)
          echo "options:"
          echo "-h, --help  show brief help"
          echo "-b, --branch_name=BRANCH_NAME specify a branch_name to use"
          echo "-g, --gpu specify to use gpu"
          exit 0
          ;;
        -b|--branch_name)
          if test $
          then
            BRANCH_NAME="$2"
            shift
          else
            echo "Error! branch_name not provided"
            exit 1
          fi
          shift
          ;;
        -g|--gpu)
          MACHINE=gpu
          shift
          ;;
    esac
done

cd docker
rm -rf serve
git clone https://github.com/pytorch/serve.git
git checkout $BRANCH_NAME
docker build --file Dockerfile.$MACHINE -t torchserve:1.0 .
