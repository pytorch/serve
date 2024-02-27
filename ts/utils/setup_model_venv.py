# This script enables creation of the model virtual environment whose path is passed in as an argument.
# The virtual environment inherits the site-packages directories of the environment from which this script is run.

import glob
import os
import site
import sys
import venv


def create_venv(venv_path):
    venv.create(venv_path, clear=True, with_pip=True)


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

    # Create a .pth file with site-packages directories to inherit, in the target venv site-packages directory
    # Ref: https://docs.python.org/3/library/site.html#module-site
    with open(
        os.path.join(target_venv_glob_matches[0], "inherited-site-packages.pth"), "w"
    ) as f:
        if site.ENABLE_USER_SITE:
            user_site_packages_dir = site.getusersitepackages()
            if os.path.exists(user_site_packages_dir):
                f.write(f"{user_site_packages_dir}\n")
                print(user_site_packages_dir)

        for global_site_packages_dir in site.getsitepackages():
            if os.path.exists(global_site_packages_dir):
                f.write(f"{global_site_packages_dir}\n")
                print(global_site_packages_dir)


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), f"{__file__} expects one argument: path to venv that should be created, but got {sys.argv}"
    create_venv(sys.argv[1])
    inherit_site_packages(sys.argv[1])
