FROM pytorch/torchserve-kfs-nightly:latest-gpu
USER root
RUN pip uninstall torchtext torchdata torch torchvision torchaudio -y
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --ignore-installed && pip install sentencepiece chardet requests
USER model-server
