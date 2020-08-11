#!/bin/bash
set -euxo pipefail

source scripts/install_utils

cleanup()
{
  stop_torchserve

  rm -rf model_store

  rm -rf logs

  # clean up residual from model-archiver IT suite.
  rm -rf model_archiver/model-archiver/htmlcov_ut model_archiver/model-archiver/htmlcov_it
}

set +u
install_torch_deps $1
set -u

if is_gpu_instance;
then
    export MKL_THREADING_LAYER=GNU
    cuda_status=$(python -c "import torch; print(int(torch.cuda.is_available()))")
    if [ $cuda_status -eq 0 ] ;
    then
      echo Ohh Its NOT running on GPU!!
      exit 1
    fi
fi

command -v "nvidia-smi" >/dev/null 2>&1

nvidia-smi