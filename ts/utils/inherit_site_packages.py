# This script enables the virtual environment that is passed in as an argument to inherit
# site-packages directories of the environment from which this script is run

import glob
import os
import site
import sys


def inherit_site_packages(venv_path):
    # Identify target venv site-packages directory
    target_venv_glob_matches = glob.glob(
        os.path.join(
            venv_path,
            "lib",
            f"python{sys.version_info[0]}.{sys.version_info[1]}",
            "site-packages",
        )
    )
    assert (
        len(target_venv_glob_matches) == 1
    ), f"{__file__} expected to find one supported python version in venv {venv_path} but found: {target_venv_glob_matches}"

    # Create sitecustomize.py in target venv site-packages directory
    # Ref: https://docs.python.org/3/library/site.html#module-sitecustomize
    with open(os.path.join(target_venv_glob_matches[0], "sitecustomize.py"), "w") as f:
        f.write("import site\n\n")
        for site_packages_dir in site.getsitepackages():
            f.write(f'site.addsitedir("{site_packages_dir}")\n')
            print(site_packages_dir)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), f"{__file__} expects one argument: path to venv that should inherit site-packages of the current venv but got {sys.argv}"
    inherit_site_packages(sys.argv[1])
