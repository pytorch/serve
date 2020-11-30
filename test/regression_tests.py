import sys
import os
import print_env_info

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts.install_from_src import install_from_src
from ts_scripts.test_regression import test_regression
from ts_scripts.test_api import test_api

from pygit2 import Repository
git_branch = Repository('.').head.shorthand
print_env_info.main(git_branch)

# Install from source
install_from_src()

# Run newman api tests
test_api("all") #"all" > management, inference, increased_timeout_inference, https collections

# Run regression tests
test_regression()
