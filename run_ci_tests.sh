#!/usr/bin/env bash
#
# A shell script to build MMS locally.
# Developer should make sure local build passed before submit PR.
# TODO: TORCHSERVE - Modify the references of awsdeeplearningteam/ to another dockerhub account, where the torchserve
#   artifacts are going to be hosted.

set -e

which docker
if [ $? -ne 0 ]
then
    echo "Please install docker."
    exit 1
fi

MMS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
BUILDSPEC="ci/buildspec.yml"

docker pull amazon/aws-codebuild-local:latest
docker pull awsdeeplearningteam/mms-build:python2.7@sha256:2b743d6724dead806873cce1330f7b8a0197399a35af47dfd7035251fdade122
docker pull awsdeeplearningteam/mms-build:python3.6@sha256:2c1afa8834907ceec641d254dffbf4bcc659ca2d00fd6f2872d7521f32c9fa2e

find . -name __pycache__ | xargs rm -rf

docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock -e "IMAGE_NAME=awsdeeplearningteam/mms-build:python2.7" -e "ARTIFACTS=${MMS_HOME}/build/artifacts2.7" -e "SOURCE=${MMS_HOME}" -e "BUILDSPEC=${BUILDSPEC}" amazon/aws-codebuild-local

find . -name __pycache__ | xargs rm -rf

docker run -it --rm -v /var/run/docker.sock:/var/run/docker.sock -e "IMAGE_NAME=awsdeeplearningteam/mms-build:python3.6" -e "ARTIFACTS=${MMS_HOME}/build/artifacts3.6" -e "SOURCE=${MMS_HOME}" -e "BUILDSPEC=${BUILDSPEC}" amazon/aws-codebuild-local
