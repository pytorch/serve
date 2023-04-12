#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

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

assert len(images_to_test) !=0, "No images to test were detected"
assert len(images_to_test) == len(tags_to_test), "number of images_to_test does not match tags_to_test"

digests = [img["digest"] for img in images_to_test]

assert len(set(digests)) == 1, f"There should be only 1 digest, found these: {set(digests)}"

print(f"Test successfull! All flags orders lead to the same image build with digest {set(digests)}")
EOF

rm -f "${IMGS_FILE}"
docker system prune -f
