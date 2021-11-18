#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ ! "$(black --version)" ]
then
    echo "Please install black."
    exit 1
fi
if [ ! "$(usort --version)" ]
then
    echo "Please install usort."
    exit 1
fi
if [ ! "$(flake8 --version)" ]
then
    echo "Please install flake8."
    exit 1
fi

# cd to the project directory
cd "$(dirname "$0")/.." || exit 1

GIT_URL_1="https://github.com/pytorch/serve.git"
GIT_URL_2="git@github.com:pytorch/serve.git"

UPSTREAM_URL="$(git config remote.upstream.url)" || true

if [ -z "$UPSTREAM_URL" ]
then
    echo "Setting upstream remote to $GIT_URL_1"
    git remote add upstream "$GIT_URL_1"
elif [ "$UPSTREAM_URL" != "$GIT_URL_1" ] && \
     [ "$UPSTREAM_URL" != "$GIT_URL_2" ]
then
    echo "upstream remote set to $UPSTREAM_URL."
    echo "Please delete the upstream remote or set it to $GIT_URL_1 to use this script."
    exit 1
fi

git fetch upstream

CHANGED_FILES="$(git diff --diff-filter=ACMRT --name-only upstream/master | grep '\.py$' | tr '\n' ' ')"

if [ "$CHANGED_FILES" != "" ]
then
    # Processing files one by one since passing directly $CHANGED_FILES will
    # treat the whole variable as a single file.
    echo "Running linters ..."
    for file in $CHANGED_FILES
    do
        echo "Checking $file"
        usort format "$file"
        black "$file" -q
        flake8 "$file" || LINT_ERRORS=1
    done
else
    echo "No changes made to any Python files. Nothing to do."
    exit 0
fi

if [ "$LINT_ERRORS" != "" ]
then
    echo "One of the linters returned an error. See above output."
    # need this so that CI fails
    exit 1
fi

# Check if any files were modified by running usort + black
# If so, then the files were formatted incorrectly (e.g. did not pass lint)
CHANGED_FILES="$(git diff --name-only | grep '\.py$' | tr '\n' ' ')"
if [ "$CHANGED_FILES" != "" ]
then
    git diff --name-only
    echo "There are uncommitted changes on disk likely caused by the linters."
    # need this so that CI fails
    exit 1
fi