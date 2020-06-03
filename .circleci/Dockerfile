FROM ubuntu:18.04 as dev
RUN apt-get update && apt-get install -y openjdk-11-jdk
COPY --from=continuumio/miniconda3:4.8.2 /opt/conda /opt/conda
ENV PATH="/opt/conda/bin/:${PATH}"
ARG PYTORCH_CHANNEL=pytorch
RUN conda install -c "${PYTORCH_CHANNEL}" -c powerai -y pytorch torchvision torchtext cpuonly psutil
