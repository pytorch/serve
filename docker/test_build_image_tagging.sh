#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

# This test checks the parsing and handling of arguments in `build_image.sh`,
# making sure that `build_image.sh` is invariant to the order of the passed 
# arguments `-py` (python version), `-t` (image tag) and `-g` (use gpu flag)
# and that tagging works properly.
# That means, we have 3 args, so there are 6 possibilities to order them and
# we expect these script runs to produce the *very same output*:
# 
# $ ./build_image.sh -py "${VERSION}" -t "${TAG}" -g
# $ ./build_image.sh -py "${VERSION}" -g -t "${TAG}"
# $ ./build_image.sh -t "${TAG}" -py "${VERSION}" -g
# $ ./build_image.sh -t "${TAG}" -g -py "${VERSION}" 
# $ ./build_image.sh -g -py "${VERSION}" -t "${TAG}"
# $ ./build_image.sh -g -t "${TAG}" -py "${VERSION}"
# 
# In order to assert the equivalence of all these variations, we take advantage
# of how docker builds images: If two images are exactly the same (ie, they are
# composed of the very same layers) they will have the same digest (ie, a hash 
# value representing the content of the image), regardless of the tag assigned 
# to the image. So, for example, if we run (with the same Dockerfile):
# 
# $ docker build -f Dockerfile -t Org/Repo:TagOne .
# $ docker build -f Dockerfile -t Org/Repo:TagTwo .
# $ docker images --no-trunc
# 
# we will see something like this:
# 
# REPOSITORY    TAG             IMAGE ID                                                               CREATED        SIZE                                   
# Org/Repo     TagOne    sha256:e3824d794c0ccf10d2f61291f34e0d7e1e02e30b3d459465bc57d04dd3b65884    30 seconds ago   2.14GB
# Org/Repo     TagTwo    sha256:e3824d794c0ccf10d2f61291f34e0d7e1e02e30b3d459465bc57d04dd3b65884    30 seconds ago   2.14GB
# 
# Notice that IMAGEID and CREATED are the same, since the first time it is 
# actually created while the second time it just uses the cached layers. 
# So the tag is "just a label" attached to the underlying image.
#
# Putting all together for our test:
# We run `build_image.sh` (on the same machine to allow docker cache) with each
# args order variation, tagging each variation with a different name (ensured 
# by the random part of the string).
# We expect:
#   - All the tags to exist (tagging works): len(images_to_test) == len(tags_to_test)
#   - All tagged images to be actually one and the same under the hood: len(set(digests)) == 1


PY_VERSION=$1
TAG_1="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"
TAG_2="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"
TAG_3="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"
TAG_4="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"
TAG_5="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"
TAG_6="org/repo:image-${PY_VERSION}-${RANDOM}-${RANDOM}-${RANDOM}-${RANDOM}"

# Do builds alternating the flags order (-g, -t, -py)
# (which should build only one underlying image)
./build_image.sh -py "${PY_VERSION}" -t "${TAG_1}" -g
./build_image.sh -py "${PY_VERSION}" -g -t "${TAG_2}"

./build_image.sh -g -py "${PY_VERSION}" -t "${TAG_3}"
./build_image.sh -g -t "${TAG_4}" -py "${PY_VERSION}"

./build_image.sh -t "${TAG_5}" -py "${PY_VERSION}" -g 
./build_image.sh -t "${TAG_6}" -g -py "${PY_VERSION}"

# Collect all the images with their tags and ids
IMGS_FILE="test_images.json"
docker images --no-trunc --format "{{json .}}" | jq '{"repo": .Repository, "tag": .Tag, "digest": .ID}' | jq -s > "${IMGS_FILE}"

python <<EOF
import json

tags_to_test = [
  "${TAG_1}",
  "${TAG_2}",
  "${TAG_3}",
  "${TAG_4}",
  "${TAG_5}",
  "${TAG_6}",
]

with open("${IMGS_FILE}") as file:
    images_to_test = [
        img
        for img in json.load(file)
        if f'{img["repo"]}:{img["tag"]}' in tags_to_test
    ]

if len(images_to_test) == 0:
    raise ValueError("No images to test were detected")

if len(images_to_test) != len(tags_to_test):
    raise ValueError(f"number of images_to_test {len(images_to_test)} does not match number of tags_to_test {len(tags_to_test)}")

digests = set(img["digest"] for img in images_to_test)

if len(digests) != 1:
    raise ValueError(f"There should be only 1 digest, found these: {digests}")

print(f"Test successfull! All flags orders lead to the same image build with digest {digests} \n")
EOF

rm -f "${IMGS_FILE}"
docker system prune -f
