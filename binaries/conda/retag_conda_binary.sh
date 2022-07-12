#!/usr/bin/env bash

# Copy pasted from script by @seemethere https://github.com/seemethere/dotfiles/blob/a0ff2bb869abc5c7641f03f196b03c6a81142c2b/bin/retag_pypi_binary.sh
# And adapted to work with conda bz2

# Usage is:
# $ NEW_VERSION="my_cool_new_version" retag_pypi_binary.sh <path_to_whl_file> <path_to_multiple_bz2_files>

# Will output a bz2 in your current directory

# TODO: Fix date suffix removal here with grep 
# TODO: Platform is unknown by using this technique
# Question: Should this script work on windows or can I rename windows binaries with no issues?

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
rm -rf lib/ bin/ info/ 
trap 'rm -rf ${tmp_dir}' EXIT

for whl_file in "$@"; do
    whl_file=$(realpath "${whl_file}")
    whl_dir="${tmp_dir}/$(basename "${whl_file}")_unzipped"
    mkdir -pv "${whl_dir}"
    (
        set -x
        tar -xf "${whl_file}" --directory "${whl_dir}"
    )
    original_version_with_date=$(grep '^Version:' "${whl_dir}"/lib/python*/site-packages/torch*/METADATA | cut -d' ' -f2)

    # # Strip extra binary information
    b_suffix="b*"
    original_version=${original_version_with_date/${b_suffix}/}
    
    # Remove all suffixed +bleh versions
    
    new_whl_file=${whl_file/${original_version}/${NEW_VERSION}}
    dist_info_folder=$(find "${whl_dir}" -type d -name '*.dist-info' | head -1)

    # Remove dev and date suffix from nightly build
    nightly_suffix=".dev[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
    new_whl_file=${OUTPUT_DIR}/$(basename "${new_whl_file/${nightly_suffix}/}")
    
    basename_dist_info_folder=$(basename "${dist_info_folder}")
    dirname_dist_info_folder=$(dirname "${dist_info_folder}")
    (
        set -x
        find "${dist_info_folder}" -type f -exec sed -i "s!${original_version}!${NEW_VERSION}!" {} \;
        # Moves distinfo from one with a version suffix to one without
        # Example: torch-1.8.0+cpu.dist-info => torch-1.8.0.dist-info
        mv "${dist_info_folder}" "${dirname_dist_info_folder}/${basename_dist_info_folder/${original_version_with_date}/${NEW_VERSION}}"
        cd "${whl_dir}"
        tar -cjSf "${new_whl_file}" .
    )
done