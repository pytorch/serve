import sys
import os

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from scripts.install_from_src import install_from_src
from scripts.test_regression import test_regression
from scripts.test_api import test_api

# Install from source
install_from_src()

# Run newman api tests
# test_api("all") #"all" > management, inference, increased_timeout_inference, https collections

# Run regression tests
test_regression()
