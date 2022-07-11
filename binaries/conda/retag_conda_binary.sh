#!/usr/bin/env bash

# Copy pasted from script by @seemethere https://github.com/seemethere/dotfiles/blob/a0ff2bb869abc5c7641f03f196b03c6a81142c2b/bin/retag_pypi_binary.sh

# Usage is:
# $ NEW_VERSION="my_cool_new_version" retag_pypi_binary.sh <path_to_whl_file> <path_to_multiple_whl_files>

# Will output a whl in your current directory

set -eou pipefail
set -ex
shopt -s globstar

OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)}

NEW_VERSION=${NEW_VERSION:-}

if [[ -z ${NEW_VERSION} ]]; then
    echo "ERROR: Environment variable NEW_VERSION must be set"
    exit 1
fi


tmp_dir="$(mktemp -d)"
trap 'rm -rf ${tmp_dir}' EXIT

for whl_file in "$@"; do
    whl_file=$(realpath "${whl_file}")
    whl_dir="${tmp_dir}/$(basename "${whl_file}")_unzipped"
    mkdir -pv "${whl_dir}"
    (
        set -x
        tar -xf "${whl_file}" --directory "${whl_dir}"
    )
    original_version=$(grep '^Version:' "${whl_dir}"/lib/python*/site-packages/torch*/METADATA | cut -d' ' -f2)
    
    # Remove all suffixed +bleh versions
    new_whl_file=${OUTPUT_DIR}/$(basename "${whl_file/${original_version}/${NEW_VERSION}}")
    dist_info_folder=$(find "${whl_dir}" -type d -name '*.dist-info' | head -1)

    # Remove _nightly suffix from package name
    nightly_suffix=".dev[0-9]*"
    new_whl_file=${OUTPUT_DIR}/$(basename "${new_whl_file/${nightly_suffix}/}")
    
    basename_dist_info_folder=$(basename "${dist_info_folder}")
    dirname_dist_info_folder=$(dirname "${dist_info_folder}")
    (
        set -x
        find "${dist_info_folder}" -type f -exec sed -i "s!${original_version}!${NEW_VERSION}!" {} \;
        # Moves distinfo from one with a version suffix to one without
        # Example: torch-1.8.0+cpu.dist-info => torch-1.8.0.dist-info
        mv "${dist_info_folder}" "${dirname_dist_info_folder}/${basename_dist_info_folder/${original_version}/${NEW_VERSION}}"
        cd "${whl_dir}"
        tar -cjvf "${new_whl_file}" .
    )
done