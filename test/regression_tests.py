from scripts.install_from_src import install_from_src
from scripts.test_regression import generate_densenet_test_model_archive, test_regression
from scripts.test_api import test_api

# Install from src
install_from_src()

# generate_densenet_test_model_archive
generate_densenet_test_model_archive()

# run_postman_test
test_api("all") #"all" > management, inference, increased_timeout_inference, https collections

# run_pytest
test_regression()
