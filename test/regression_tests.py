import sys
import os

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.install_from_src import install_from_src
from ts_scripts.regression_utils import test_regression
from ts_scripts.api_utils import test_api
from ts_scripts import print_env_info  as build_hdr_printer

from pygit2 import Repository
git_branch = Repository('.').head.shorthand
build_hdr_printer.main(git_branch)

# Install from source
install_from_src()

# Run newman api tests
test_api("all") #"all" > management, inference, increased_timeout_inference, https collections

# Run regression tests
test_regression()
