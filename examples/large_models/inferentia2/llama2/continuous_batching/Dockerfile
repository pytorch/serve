FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04
# workaround transformers 4.35 failure of downloading HF model bin files
RUN pip install transformers==4.34
WORKDIR /home/model-server
RUN git clone https://github.com/pytorch/serve.git \
    && cd serve
WORKDIR /home/model-server/serve
RUN git checkout feat/inf2_cb
RUN pip install pygit2
RUN python ts_scripts/install_from_src.py
