import sys
import os
from pygit2 import Repository

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.install_from_src import install_from_src
from ts_scripts.regression_utils import test_regression
from ts_scripts.api_utils import test_api
from ts_scripts import print_env_info as build_hdr_printer
from ts_scripts.utils import check_python_version
from ts_scripts import marsgen as mg

import datetime

now = datetime.datetime.now()
print("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))

check_python_version()

git_branch = Repository('.').head.shorthand
build_hdr_printer.main(git_branch)

# Install from source
install_from_src()

# Generate mar file
mg.generate_mars()

# Run newman api tests
test_api(
    "all"
)  #"all" > management, inference, increased_timeout_inference, https collections

# Run regression tests
test_regression()

# delete mar_gen_dir
mg.delete_model_store_gen_dir()