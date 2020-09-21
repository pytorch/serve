import glob
import os
import platform
import sys
import time
import argparse

TEST_DIR = os.path.join("test")
MODEL_STORE_DIR = os.path.join("model_store")

ARTIFACTS_MANAGEMENT_DIR = os.path.join("artifacts", "management")
ARTIFACTS_INFERENCE_DIR = os.path.join("artifacts", "inference")
ARTIFACTS_HTTPS_DIR = os.path.join("artifacts", "https")

TS_CONSOLE_LOG_FILE = os.path.join("ts_console.log")
TS_CONFIG_FILE_HTTPS = os.path.join("resources", "config.properties")

POSTMAN_ENV_FILE = os.path.join("postman", "environment.json")
POSTMAN_COLLECTION_MANAGEMENT = os.path.join("postman", "management_api_test_collection.json")
POSTMAN_COLLECTION_INFERENCE = os.path.join("postman", "inference_api_test_collection.json")
POSTMAN_COLLECTION_HTTPS = os.path.join("postman", "https_test_collection.json")
POSTMAN_DATA_FILE_INFERENCE = os.path.join("postman", "inference_data.json")

REPORT_FILE = os.path.join("report.html")


torchserve_command = {
    "Windows": "torchserve.exe",
    "Darwin": "torchserve",
    "Linux": "torchserve"
}

def start_ts(model_store_dir, log_file):
    os.system(f"{torchserve_command[platform.system()]} --ncs --start --model-store {model_store_dir} >> {log_file} 2>&1 &")
    time.sleep(10)


def start_ts_secure(model_store_dir, log_file, ts_https_config_file):
    os.system(f"{torchserve_command[platform.system()]} --ncs --start --ts-config {ts_https_config_file} --model-store {model_store_dir} >> {log_file} 2>&1 &")
    time.sleep(10)


def stop_ts(log_file):
    os.system(f"{torchserve_command[platform.system()]} --stop >> {log_file} 2>&1")
    time.sleep(10)


def cleanup_model_store():
    # rm -rf $MODEL_STORE_DIR / *
    for f in glob.glob(os.path.join(MODEL_STORE_DIR, "*")):
        os.remove(f)


def move_logs(log_file, artifact_dir):
    logs_dir = os.path.join("logs")
    os.rename(log_file, os.path.join(logs_dir, log_file))    # mv file logs/
    os.rename(logs_dir, os.path.join(artifact_dir, logs_dir)) # mv logs/ dir


def trigger_management_tests():
    """ Return exit code of newman execution of management collection """
    start_ts(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_MANAGEMENT} -r cli,html --reporter-html-export {ARTIFACTS_MANAGEMENT_DIR}/{REPORT_FILE} --verbose")
    stop_ts(TS_CONSOLE_LOG_FILE)
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_MANAGEMENT_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_inference_tests():
    """ Return exit code of newman execution of inference collection """
    start_ts(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE} -d {POSTMAN_DATA_FILE_INFERENCE} -r cli,html --reporter-html-export {ARTIFACTS_INFERENCE_DIR}/{REPORT_FILE} --verbose")
    stop_ts(TS_CONSOLE_LOG_FILE)
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_INFERENCE_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_https_tests():
    """ Return exit code of newman execution of https collection """
    start_ts_secure(MODEL_STORE_DIR, TS_CONSOLE_LOG_FILE, TS_CONFIG_FILE_HTTPS)
    EXIT_CODE = os.system(f"newman run --insecure -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_HTTPS} -r cli,html --reporter-html-export {ARTIFACTS_HTTPS_DIR}/{REPORT_FILE} --verbose")
    stop_ts(TS_CONSOLE_LOG_FILE)
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_HTTPS_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_all():
    exit_code1 = trigger_management_tests()
    exit_code2 = trigger_inference_tests()
    exit_code3 = trigger_https_tests()
    return 1 if any(code != 0 for code in [exit_code1, exit_code2, exit_code3]) else 0


os.chdir(TEST_DIR)
for DIR in [MODEL_STORE_DIR, ARTIFACTS_MANAGEMENT_DIR, ARTIFACTS_INFERENCE_DIR, ARTIFACTS_HTTPS_DIR] :
    os.makedirs(DIR, exist_ok=True)


parser = argparse.ArgumentParser(description="Execute newman API test suite")
parser.add_argument("collection", type=str, help="Collection Name")
args = parser.parse_args()

collection = args.collection

switcher = {
    "management": trigger_management_tests,
    "inference": trigger_inference_tests,
    "https": trigger_https_tests,
    "all": trigger_all
}

sys.exit(switcher[collection]())