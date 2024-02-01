FROM python:3.13.0a3-slim
ARG BRANCH_NAME_KF=master

RUN apt-get update \
  && apt-get install -y --no-install-recommends git

COPY . .
RUN pip install --upgrade pip 
RUN git clone -b ${BRANCH_NAME_KF} https://github.com/kserve/kserve 
RUN pip install -e ./kserve/python/kserve
ENTRYPOINT ["python", "-m", "image_transformer"]
