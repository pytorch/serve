import os
import sys

import pytest
import test_utils
import torch

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
)
sys.path.append(REPO_ROOT)

print(REPO_ROOT)
TEST_DIR = os.path.join(REPO_ROOT, "test")
MODEL_STORE_DIR = os.path.join("model_store")

POSTMAN_LARGE_MODEL_INFERENCE_DATA_FILE = os.path.join(
    "postman", "large_model_inference_data.json"
)
TS_CONSOLE_LOG_FILE = os.path.join("ts_log.log")
POSTMAN_ENV_FILE = os.path.join("postman", "environment.json")
POSTMAN_COLLECTION_INFERENCE = os.path.join(
    "postman", "inference_api_test_collection.json"
)
ARTIFACTS_INFERENCE_DIR = os.path.join("artifacts", "inference")
REPORT_FILE = os.path.join("report.html")


@pytest.mark.skipif(
    not ((torch.cuda.device_count() > 0) and torch.cuda.is_available()),
    reason="Test to be run on GPU only",
)
def test_large_model_inference():
    """Run a Newman test for distributed inference on a large model"""
    os.chdir(TEST_DIR)

    test_utils.start_torchserve(model_store=MODEL_STORE_DIR, gen_mar=False)

    try:
        command = f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE} -d {POSTMAN_LARGE_MODEL_INFERENCE_DATA_FILE} -r cli,htmlextra --reporter-htmlextra-export {ARTIFACTS_INFERENCE_DIR}/{REPORT_FILE} --verbose"
        result = os.system(command)
        assert (
            result == 0
        ), "Error: Distributed inference failed, the exit code is not zero"
    finally:
        test_utils.stop_torchserve()
        test_utils.cleanup_model_store(model_store=MODEL_STORE_DIR)
