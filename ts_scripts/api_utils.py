import glob
import os
import shutil
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

from ts_scripts import tsutils as ts

TEST_DIR = os.path.join("test")
MODEL_STORE_DIR = os.path.join("model_store")

# Contains separate collection and respecive tests for both kfserving and torchserve execution

### Torchserve
ARTIFACTS_MANAGEMENT_DIR = os.path.join("artifacts", "management")
ARTIFACTS_INFERENCE_DIR = os.path.join("artifacts", "inference")
ARTIFACTS_EXPLANATION_DIR = os.path.join("artifacts", "explanation")
ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR = os.path.join("artifacts", "increased_timeout_inference")
ARTIFACTS_HTTPS_DIR = os.path.join("artifacts", "https")

TS_CONSOLE_LOG_FILE = os.path.join("ts_console.log")
TS_CONFIG_FILE_HTTPS = os.path.join("resources", "config.properties")

POSTMAN_ENV_FILE = os.path.join("postman", "environment.json")
POSTMAN_INFERENCE_DATA_FILE = os.path.join("postman", "inference_data.json")
POSTMAN_EXPLANATION_DATA_FILE = os.path.join("postman", "explanation_data.json")
POSTMAN_MANAGEMENT_DATA_FILE = os.path.join("postman", "management_data.json")
POSTMAN_INCRSD_TIMEOUT_INFERENCE_DATA_FILE = os.path.join("postman", "increased_timeout_inference.json")

#only one management collection for both kfserving and torchserve
POSTMAN_COLLECTION_MANAGEMENT = os.path.join("postman", "management_api_test_collection.json")
POSTMAN_COLLECTION_INFERENCE = os.path.join("postman", "inference_api_test_collection.json")
POSTMAN_COLLECTION_EXPLANATION = os.path.join("postman", "explanation_api_test_collection.json")

POSTMAN_COLLECTION_HTTPS = os.path.join("postman", "https_test_collection.json")


### KFserving
ARTIFACTS_MANAGEMENT_DIR_KF = os.path.join("artifacts", "management_kf")
ARTIFACTS_INFERENCE_DIR_KF = os.path.join("artifacts", "inference_kf")
ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR_KF = os.path.join("artifacts", "increased_timeout_inference_kf")
ARTIFACTS_HTTPS_DIR_KF = os.path.join("artifacts", "https_kf")

TS_CONFIG_FILE_HTTPS_KF = os.path.join("resources", "config_kf.properties")

POSTMAN_INFERENCE_DATA_FILE_KF = os.path.join("postman", "kf_inference_data.json")
POSTMAN_INCRSD_TIMEOUT_INFERENCE_DATA_FILE_KF = os.path.join("postman", "increased_timeout_inference.json")

POSTMAN_COLLECTION_INFERENCE_KF = os.path.join("postman", "kf_api_test_collection.json")

POSTMAN_COLLECTION_HTTPS_KF = os.path.join("postman", "kf_https_test_collection.json")

REPORT_FILE = os.path.join("report.html")


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
    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_MANAGEMENT} -d {POSTMAN_MANAGEMENT_DATA_FILE} -r cli,html --reporter-html-export {ARTIFACTS_MANAGEMENT_DIR}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_MANAGEMENT_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_inference_tests():
    """ Return exit code of newman execution of inference collection """
    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE} -d {POSTMAN_INFERENCE_DATA_FILE} -r cli,html --reporter-html-export {ARTIFACTS_INFERENCE_DIR}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_INFERENCE_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_explanation_tests():
    """ Return exit code of newman execution of inference collection """
    
    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_EXPLANATION} -d {POSTMAN_EXPLANATION_DATA_FILE} -r cli,html --reporter-html-export {ARTIFACTS_INFERENCE_DIR}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_EXPLANATION_DIR)
    cleanup_model_store()
    return EXIT_CODE


def trigger_incr_timeout_inference_tests():
    """ Return exit code of newman execution of increased timeout inference collection """

    # Configuration with increased timeout
    config_file = open("config.properties", "w")
    config_file.write("default_response_timeout=300")
    config_file.close()

    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, config_file="config.properties", log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE} -d {POSTMAN_INCRSD_TIMEOUT_INFERENCE_DATA_FILE} -r cli,html --reporter-html-export {ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR)
    cleanup_model_store()

    os.remove("config.properties")
    return EXIT_CODE


def trigger_https_tests():
    """ Return exit code of newman execution of https collection """
    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, config_file=TS_CONFIG_FILE_HTTPS, log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run --insecure -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_HTTPS} -r cli,html --reporter-html-export {ARTIFACTS_HTTPS_DIR}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_HTTPS_DIR)
    cleanup_model_store()
    return EXIT_CODE

## KFserving tests starts here
def trigger_management_tests_kf():
    """ Return exit code of newman execution of management collection """
    
    config_file = open("config.properties", "w")
    config_file.write("service_envelope=kfserving")
    config_file.close()

    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, config_file="config.properties", log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_MANAGEMENT} -d {POSTMAN_MANAGEMENT_DATA_FILE} -r cli,html --reporter-html-export {ARTIFACTS_MANAGEMENT_DIR_KF}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_MANAGEMENT_DIR_KF)
    cleanup_model_store()
    os.remove("config.properties")
    return EXIT_CODE

def trigger_inference_tests_kf():
    """ Return exit code of newman execution of inference collection """

    config_file = open("config.properties", "w")
    config_file.write("service_envelope=kfserving")
    config_file.close()

    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, config_file="config.properties", log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_INFERENCE_KF} -d {POSTMAN_INFERENCE_DATA_FILE_KF} -r cli,html --reporter-html-export {ARTIFACTS_INFERENCE_DIR_KF}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_INFERENCE_DIR_KF)
    cleanup_model_store()
    os.remove("config.properties")
    return EXIT_CODE


def trigger_https_tests_kf():
    """ Return exit code of newman execution of https collection """
    ts.start_torchserve(ncs=True, model_store=MODEL_STORE_DIR, config_file=TS_CONFIG_FILE_HTTPS_KF, log_file=TS_CONSOLE_LOG_FILE)
    EXIT_CODE = os.system(f"newman run --insecure -e {POSTMAN_ENV_FILE} {POSTMAN_COLLECTION_HTTPS_KF} -r cli,html --reporter-html-export {ARTIFACTS_HTTPS_DIR_KF}/{REPORT_FILE} --verbose")
    ts.stop_torchserve()
    move_logs(TS_CONSOLE_LOG_FILE, ARTIFACTS_HTTPS_DIR_KF)
    cleanup_model_store()
    return EXIT_CODE


def trigger_all():
    exit_code1 = trigger_management_tests()
    exit_code2 = trigger_inference_tests()
    exit_code3 = trigger_incr_timeout_inference_tests()
    exit_code4 = trigger_https_tests()
    exit_code5 = trigger_management_tests_kf()
    exit_code6 = trigger_inference_tests_kf()
    exit_code7 = trigger_https_tests_kf()
    exit_code8 = trigger_explanation_tests()
    return 1 if any(code != 0 for code in [exit_code1, exit_code2, exit_code3, exit_code4, exit_code5, exit_code6, exit_code7, exit_code8]) else 0


def test_api(collection):
    os.chdir(TEST_DIR)
    ALL_DIRS = [MODEL_STORE_DIR, ARTIFACTS_MANAGEMENT_DIR, ARTIFACTS_INFERENCE_DIR, ARTIFACTS_EXPLANATION_DIR,
    ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR, ARTIFACTS_HTTPS_DIR, ARTIFACTS_MANAGEMENT_DIR_KF, ARTIFACTS_INFERENCE_DIR_KF, 
    ARTIFACTS_INCRSD_TIMEOUT_INFERENCE_DIR_KF, ARTIFACTS_HTTPS_DIR_KF] 

    for DIR in ALL_DIRS:
        shutil.rmtree(DIR, True)
        os.makedirs(DIR, exist_ok=True)

    switcher = {
        "management": trigger_management_tests,
        "management_kf": trigger_management_tests_kf,
        "inference": trigger_inference_tests,
        "inference_kf": trigger_inference_tests_kf,
        "explanation": trigger_explanation_tests,
        "increased_timeout_inference": trigger_incr_timeout_inference_tests,
        "https": trigger_https_tests,
        "https_kf": trigger_https_tests_kf,
        "all": trigger_all
    }

    exit_code = switcher[collection]()
    os.chdir(REPO_ROOT)

    if exit_code != 0:
        sys.exit("## Newman API Tests Failed !")
