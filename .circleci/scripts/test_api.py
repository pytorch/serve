import glob
import os
import sys
import time
import argparse

TEST_DIR = os.path.join("test")
MODEL_STORE_DIR = os.path.join("model_store")

ARTIFACTS_BASE_DIR = os.path.join("artifacts")
ARTIFACTS_MANAGEMENT_DIR = os.path.join(ARTIFACTS_BASE_DIR, "management")
ARTIFACTS_INFERENCE_DIR = os.path.join(ARTIFACTS_BASE_DIR, "inference")
ARTIFACTS_HTTPS_DIR = os.path.join(ARTIFACTS_BASE_DIR, "https")

TS_CONSOLE_LOG_FILE = os.path.join("ts_console.log")
TS_CONFIG_FILE_HTTPS = os.path.join("resources", "config.properties")

POSTMAN_ENV_FILE = os.path.join("postman", "environment.json")
POSTMAN_COLLECTION_MANAGEMENT = os.path.join("postman", "management_api_test_collection.json")
POSTMAN_COLLECTION_INFERENCE = os.path.join("postman", "inference_api_test_collection.json")
POSTMAN_COLLECTION_HTTPS = os.path.join("postman", "https_test_collection.json")
POSTMAN_DATA_FILE_INFERENCE = os.path.join("postman", "inference_data.json")

REPORT_FILE = os.path.join("report.html")

def start_ts(model_store_dir, log_file):
    os.system(f"torchserve --ncs --start --model-store {model_store_dir} >> {log_file} 2>&1")
    time.sleep(10)


def start_ts_secure(model_store_dir, log_file, ts_https_config_file):
  os.system(f"torchserve --ncs --start --ts-config {ts_https_config_file} --model-store {model_store_dir} >> {log_file} 2>&1")
  time.sleep(10)

def stop_ts():
  os.system("torchserve --stop")
  time.sleep(10)

def cleanup_model_store():
    (os.remove(f) for f in glob.glob(os.path.join(MODEL_STORE_DIR, "*"))) # rm -rf $MODEL_STORE_DIR/*

def move_logs(file, dir):
    logs_dir = os.path.join("logs")
    os.rename(file, os.path.join(logs_dir, file))    # mv file logs/
    os.rename(logs_dir, os.path.join(dir, logs_dir)) # mv logs/ dir

def trigger_management_tests():
    """ Return exit code of newman execution of management collection """
    start_ts(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_MANAGEMENT} -r cli,html --reporter-html-export {ARTIFACTS_MANAGEMENT_DIR}/{REPORT_FILE} --verbose")
    stop_ts()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_MANAGEMENT_DIR)
    cleanup_model_store()
    return EXIT_CODE

def trigger_inference_tests():
    """ Return exit code of newman execution of inference collection """
    start_ts(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE} -d {POSTMAN_DATA_FILE_INFERENCE} -r cli,html --reporter-html-export {ARTIFACTS_INFERENCE_DIR}/{REPORT_FILE} --verbose")
    stop_ts()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_INFERENCE_DIR)
    cleanup_model_store()
    return EXIT_CODE

def trigger_https_tests():
    """ Return exit code of newman execution of https collection """
    start_ts_secure(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE, TS_CONFIG_FILE_HTTPS)
    EXIT_CODE = os.system(f"newman run --insecure -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_HTTPS} -r cli,html --reporter-html-export {ARTIFACTS_HTTPS_DIR}/{REPORT_FILE} --verbose")
    stop_ts()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_HTTPS_DIR)
    cleanup_model_store()
    return EXIT_CODE

def trigger_all():
    exit_code1 = trigger_management_tests()
    exit_code2 = trigger_inference_tests()
    exit_code3 = trigger_https_tests()
    EXIT_CODE = 1 if any(code != 0 for code in [exit_code1, exit_code2, exit_code3]) else 0
    return EXIT_CODE


os.chdir(TEST_DIR)
(os.makedirs(DIR, exist_ok=True) for DIR in [MODEL_STORE_DIR, ARTIFACTS_MANAGEMENT_DIR, ARTIFACTS_INFERENCE_DIR, ARTIFACTS_HTTPS_DIR])


parser = argparse.ArgumentParser(description="Execute newman API test suite")
parser.add_argument("collection", type=str, help="Collection Name")
args = parser.parse_args()

collection = args.workflow

switcher = {
    "management" : trigger_management_tests,
    "inference" : trigger_management_tests,
    "https" : trigger_https_tests,
    "all" : trigger_all
}

sys.exit(switcher[collection]())