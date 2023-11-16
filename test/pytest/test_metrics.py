import glob
import os
import platform
import re
import shutil
import subprocess
import time
from os import path

import requests
import test_utils

NUM_STARTUP_CFG = 0
FRONTEND_METRICS = [
    "Requests2XX",
    "Requests4XX",
    "Requests5XX",
    "ts_inference_requests_total",
    "ts_inference_latency_microseconds",
    "ts_queue_latency_microseconds",
    "QueueTime",
    "WorkerThreadTime",
    "WorkerLoadTime",
]
SYSTEM_METRICS = [
    "CPUUtilization",
    "MemoryUsed",
    "MemoryAvailable",
    "MemoryUtilization",
    "DiskUsage",
    "DiskUtilization",
    "DiskAvailable",
    "GPUMemoryUtilization",
    "GPUMemoryUsed",
    "GPUUtilization",
]
BACKEND_METRICS = [
    "HandlerTime",
    "PredictionTime",
    "InferenceRequestCount",
    "InitializeCallCount",
    "PreprocessCallCount",
    "PostprocessCallCount",
    "RequestBatchSize",
    "SizeOfImage",
]
AUTO_DETECT_BACKEND_METRICS = [
    "HandlerMethodTime",
    "ExamplePercentMetric",
]


def setup_module(module):
    test_utils.torchserve_cleanup()

    model_archiver_cmd = test_utils.model_archiver_command_builder(
        model_name="mnist_custom_metrics",
        version="1.0",
        model_file=os.path.join(
            test_utils.REPO_ROOT, "examples", "image_classifier", "mnist", "mnist.py"
        ),
        serialized_file=os.path.join(
            test_utils.REPO_ROOT,
            "examples",
            "image_classifier",
            "mnist",
            "mnist_cnn.pt",
        ),
        handler=os.path.join(
            test_utils.REPO_ROOT, "examples", "custom_metrics", "mnist_handler.py"
        ),
        export_path=test_utils.MODEL_STORE,
    )
    model_archiver_cmd = model_archiver_cmd.split(" ")
    subprocess.run(model_archiver_cmd)


def teardown_module(module):
    test_utils.torchserve_cleanup()


def logs_created(no_config_snapshots=False):
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)
    assert len(glob.glob("logs/access_log.log")) == 1
    assert len(glob.glob("logs/model_log.log")) == 1
    assert len(glob.glob("logs/ts_log.log")) == 1


def validate_metrics_created(no_config_snapshots=False):
    test_utils.delete_all_snapshots()
    global NUM_STARTUP_CFG
    # Reset NUM_STARTUP_CFG as we are deleting snapshots in the previous step
    NUM_STARTUP_CFG = 0
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)
    if not no_config_snapshots:
        NUM_STARTUP_CFG += 1

    assert len(glob.glob("logs/model_metrics.log")) == 1
    assert len(glob.glob("logs/ts_metrics.log")) == 1


def run_log_location_var(custom_path=test_utils.ROOT_DIR, no_config_snapshots=False):
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)

    # This check warrants that we are not accidentally monitoring a readonly logs/snapshot directory
    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + "/access_log.log")) == 1
        assert len(glob.glob(custom_path + "/model_log.log")) == 1
        assert len(glob.glob(custom_path + "/ts_log.log")) == 1


def register_model_and_make_inference_request():
    test_utils.register_model("mnist_custom_metrics", "mnist_custom_metrics.mar")
    data_file = os.path.join(
        test_utils.REPO_ROOT,
        "examples",
        "image_classifier",
        "mnist",
        "test_data",
        "0.png",
    )
    with open(data_file, "rb") as input_data:
        requests.post(
            url=f"http://localhost:8080/predictions/mnist_custom_metrics",
            data=input_data,
        )


def validate_metrics_log(log_filename, metric_names, present):
    assert len(glob.glob("logs/" + log_filename)) == 1
    metrics_path = glob.glob("logs/" + log_filename)[0]
    if present:
        assert os.path.getsize(metrics_path) > 0

    metrics_regex = re.compile("|".join(metric_names), flags=re.IGNORECASE)
    with open(metrics_path, "rt") as metrics_file:
        metrics_data = metrics_file.read()
        matched_metrics = re.findall(metrics_regex, metrics_data)

    if present:
        assert len(matched_metrics) > 0
    else:
        assert len(matched_metrics) == 0


def test_logs_created():
    logs_created()
    global NUM_STARTUP_CFG
    NUM_STARTUP_CFG = len(glob.glob("logs/config/*startup.cfg"))


def test_logs_startup_cfg_created_snapshot_enabled():
    """
    Validates that access logs are getting created correctly.
    """
    logs_created(no_config_snapshots=False)
    global NUM_STARTUP_CFG
    assert len(glob.glob("logs/config/*startup.cfg")) == NUM_STARTUP_CFG + 1
    NUM_STARTUP_CFG += 1


def test_logs_startup_cfg_created_snapshot_disabled():
    """
    Validates that access logs are getting created correctly.
    """
    logs_created(no_config_snapshots=True)
    global NUM_STARTUP_CFG
    assert len(glob.glob("logs/config/*startup.cfg")) == NUM_STARTUP_CFG


def test_metrics_startup_cfg_created_snapshot_enabled():
    """
    Validates that model metrics is getting created with snapshot enabled.
    """
    validate_metrics_created(no_config_snapshots=False)
    assert len(glob.glob("logs/config/*startup.cfg")) == NUM_STARTUP_CFG


def test_metrics_startup_cfg_created_snapshot_disabled():
    """
    Validates that model metrics is getting created with snapshot disabled.
    """
    validate_metrics_created(no_config_snapshots=True)
    assert len(glob.glob("logs/config/*startup.cfg")) == 0


def test_log_location_var_snapshot_disabled():
    """
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    test_utils.stop_torchserve()
    os.environ["LOG_LOCATION"] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    test_utils.stop_torchserve()
    del os.environ["LOG_LOCATION"]
    for f in glob.glob(path.join(test_utils.ROOT_DIR, "*.log")):
        print("-------------Deleting " + f)
        os.remove(f)
    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_log_location_var_snapshot_enabled():
    """
    Validates that non metrics logs get saved in directory configured via LOG_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set LOG_LOCATION in environment variable and rerun torchserve
    test_utils.stop_torchserve()
    os.environ["LOG_LOCATION"] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=False)
    requests.post("http://127.0.0.1:8081/models?url=mnist_custom_metrics.mar")
    # We stop torchserve again here so that we can remove the LOG_LOCATION setting from environment variable
    test_utils.stop_torchserve()
    print("Waiting to stop")
    time.sleep(15)
    del os.environ["LOG_LOCATION"]

    # In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob(path.join(test_utils.ROOT_DIR, "config/*startup.cfg"))) >= 1
    if platform.system() != "Windows":
        assert (
            len(glob.glob(path.join(test_utils.ROOT_DIR, "config/*shutdown.cfg"))) >= 1
        )
    assert len(glob.glob(path.join(test_utils.ROOT_DIR, "config/*snap*.cfg"))) >= 1
    for f in glob.glob(path.join(test_utils.ROOT_DIR, "*.log")):
        print("-------------Deleting " + f)
        os.remove(f)

    shutil.rmtree(path.join(test_utils.ROOT_DIR, "config"))

    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_async_logging():
    """Validates that we can use async_logging flag while starting Torchserve"""
    # Need to stop torchserve as we need to check if log files get generated with 'aysnc_logging' flag
    test_utils.stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = os.path.join(test_utils.ROOT_DIR, "async-log-config.properties")
    with open(async_config_file, "w+") as f:
        f.write("async_logging=true")
    test_utils.start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob("logs/access_log.log")) == 1
    assert len(glob.glob("logs/model_log.log")) == 1
    assert len(glob.glob("logs/ts_log.log")) == 1


def test_async_logging_non_boolean():
    """Validates that Torchserve uses default value for async_logging flag
    if we assign a non boolean value to this flag"""
    test_utils.stop_torchserve()
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    # delete_all_snapshots()
    async_config_file = os.path.join(test_utils.ROOT_DIR, "async-log-config.properties")
    with open(async_config_file, "w+") as f:
        f.write("async_logging=2")
    test_utils.start_torchserve(snapshot_file=async_config_file)
    assert len(glob.glob("logs/access_log.log")) == 1
    assert len(glob.glob("logs/model_log.log")) == 1
    assert len(glob.glob("logs/ts_log.log")) == 1
    test_utils.stop_torchserve()


def run_metrics_location_var(
    custom_path=test_utils.ROOT_DIR, no_config_snapshots=False
):
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve(no_config_snapshots=no_config_snapshots)

    if os.access(custom_path, os.W_OK):
        assert len(glob.glob(custom_path + "/ts_metrics.log")) == 1
        assert len(glob.glob(custom_path + "/model_metrics.log")) == 1


def test_metrics_location_var_snapshot_disabled():
    """
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    test_utils.stop_torchserve()
    os.environ["METRICS_LOCATION"] = test_utils.ROOT_DIR
    run_metrics_location_var(no_config_snapshots=True)
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    test_utils.stop_torchserve()
    del os.environ["METRICS_LOCATION"]
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)
    # Remove any old snapshots
    test_utils.delete_all_snapshots()


def test_metrics_location_var_snapshot_enabled():
    """
    Validates that metrics related logs get saved in directory configured via METRICS_LOCATION
    environment variable.
    """
    # We stop torchserve here so that we can set METRICS_LOCATION in environment variable and
    # restart torchserve
    test_utils.stop_torchserve()
    os.environ["METRICS_LOCATION"] = test_utils.ROOT_DIR
    run_metrics_location_var(no_config_snapshots=False)
    requests.post("http://127.0.0.1:8081/models?url=mnist_custom_metrics.mar")
    # We stop torchserve again here so that we can remove the METRICS_LOCATION setting
    # from environment variable
    test_utils.stop_torchserve()
    del os.environ["METRICS_LOCATION"]
    # In case of snapshot enabled, we get these three config files additionally in the custom directory
    assert len(glob.glob("logs/config/*startup.cfg")) >= 1
    if platform.system() != "Windows":
        assert len(glob.glob("logs/config/*shutdown.cfg")) >= 1
    assert len(glob.glob("logs/config/*snap*.cfg")) >= 1
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)


def test_log_location_and_metric_location_vars_snapshot_enabled():
    """
    Validates that metrics & non metrics related logs get saved in directory configured as per
     METRICS_LOCATION & LOG_LOCATION environment variables with snaphsot enabled.
    """
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    os.environ["LOG_LOCATION"] = test_utils.ROOT_DIR
    os.environ["METRICS_LOCATION"] = test_utils.ROOT_DIR
    run_log_location_var(no_config_snapshots=False)
    run_metrics_location_var(no_config_snapshots=False)
    requests.post("http://127.0.0.1:8081/models?url=mnist_custom_metrics.mar")
    # We stop torchserve again here so that we can remove the LOG_LOCATION & METRICS_LOCATION
    # setting from environment variable
    test_utils.stop_torchserve()
    del os.environ["LOG_LOCATION"]
    del os.environ["METRICS_LOCATION"]
    assert (
        len(glob.glob(os.path.join(test_utils.ROOT_DIR, "config", "*startup.cfg"))) >= 1
    )
    if platform.system() != "Windows":
        assert (
            len(glob.glob(os.path.join(test_utils.ROOT_DIR, "config", "*shutdown.cfg")))
            >= 1
        )
    assert (
        len(glob.glob(os.path.join(test_utils.ROOT_DIR, "config", "*snap*.cfg"))) >= 1
    )
    for f in glob.glob(test_utils.ROOT_DIR + "*.log"):
        os.remove(f)

    shutil.rmtree(path.join(test_utils.ROOT_DIR, "config"))


def test_log_location_var_snapshot_disabled_custom_path_read_only():
    """
    Validates that we should not be able to create non metrics related logs if the directory configured
    via 'LOG_LOCATION' is a read only directory.
    """
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    RDONLY_DIR = path.join(test_utils.ROOT_DIR, "rdonly_dir")
    os.environ["LOG_LOCATION"] = RDONLY_DIR
    try:
        run_log_location_var(custom_path=RDONLY_DIR, no_config_snapshots=True)
        assert len(glob.glob(RDONLY_DIR + "/logs/access_log.log")) == 0
        assert len(glob.glob(RDONLY_DIR + "/logs/model_log.log")) == 0
        assert len(glob.glob(RDONLY_DIR + "/logs/ts_log.log")) == 0
        assert len(glob.glob("logs/model_metrics.log")) == 1
        assert len(glob.glob("logs/ts_metrics.log")) == 1
    finally:
        del os.environ["LOG_LOCATION"]


def test_metrics_location_var_snapshot_enabled_rdonly_dir():
    """
    Validates that we should not be able to create metrics related logs if the directory configured
    via 'METRICS_LOCATION' is a read only directory.
    """
    # Torchserve cleanup is required here as we are going to set 'LOG_LOCATION' environment variable
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # First remove existing logs otherwise it may be a false positive case
    for f in glob.glob("logs/*.log"):
        os.remove(f)
    RDONLY_DIR = path.join(test_utils.ROOT_DIR, "rdonly_dir")
    os.environ["METRICS_LOCATION"] = RDONLY_DIR
    try:
        run_metrics_location_var(custom_path=RDONLY_DIR, no_config_snapshots=False)
        requests.post("http://127.0.0.1:8081/models?url=mnist_custom_metrics.mar")
        assert len(glob.glob("logs/access_log.log")) == 1
        assert len(glob.glob("logs/model_log.log")) == 1
        assert len(glob.glob("logs/ts_log.log")) == 1
        assert len(glob.glob("logs/config/*snap*.cfg")) == 1
        assert len(glob.glob(RDONLY_DIR + "/logs/model_metrics.log")) == 0
        assert len(glob.glob(RDONLY_DIR + "/logs/ts_metrics.log")) == 0
    finally:
        del os.environ["METRICS_LOCATION"]


def test_metrics_log_mode():
    """
    Validates that metrics uses log mode by default
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT, "examples", "custom_metrics", "metrics.yaml"
    )

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("ts_metrics.log", FRONTEND_METRICS, True)
        validate_metrics_log("ts_metrics.log", SYSTEM_METRICS, True)
        validate_metrics_log("model_metrics.log", BACKEND_METRICS, True)
        validate_metrics_log("model_metrics.log", AUTO_DETECT_BACKEND_METRICS, False)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_CONFIG"]
        os.remove(config_file)


def test_metrics_prometheus_mode():
    """
    Validates metrics prometheus mode
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_METRICS_MODE"] = "prometheus"
    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT, "examples", "custom_metrics", "metrics.yaml"
    )

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("ts_metrics.log", FRONTEND_METRICS, False)
        validate_metrics_log("ts_metrics.log", SYSTEM_METRICS, False)
        validate_metrics_log("model_metrics.log", BACKEND_METRICS, False)
        validate_metrics_log("model_metrics.log", AUTO_DETECT_BACKEND_METRICS, False)

        response = requests.get("http://localhost:8082/metrics")
        prometheus_metrics = response.text
        for metric_name in FRONTEND_METRICS:
            assert metric_name in prometheus_metrics
        for metric_name in SYSTEM_METRICS:
            assert metric_name in prometheus_metrics
        for metric_name in BACKEND_METRICS:
            assert metric_name in prometheus_metrics
        for metric_name in AUTO_DETECT_BACKEND_METRICS:
            assert metric_name not in prometheus_metrics

        prometheus_metric_patterns = [
            r"TYPE Requests2XX counter",
            r'Requests2XX\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE ts_inference_requests_total counter",
            r'ts_inference_requests_total\{model_name="mnist_custom_metrics",model_version="default",hostname=".+",\} \d+\.\d+',
            r"TYPE ts_inference_latency_microseconds counter",
            r'ts_inference_latency_microseconds\{model_name="mnist_custom_metrics",model_version="default",hostname=".+",\} \d+\.\d+',
            r"TYPE ts_queue_latency_microseconds counter",
            r'ts_queue_latency_microseconds\{model_name="mnist_custom_metrics",model_version="default",hostname=".+",\} \d+\.\d+',
            r"TYPE QueueTime gauge",
            r'QueueTime\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE WorkerThreadTime gauge",
            r'WorkerThreadTime\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE WorkerLoadTime gauge",
            r'WorkerLoadTime\{WorkerName=".+",Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE CPUUtilization gauge",
            r'CPUUtilization\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE MemoryUsed gauge",
            r'MemoryUsed\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE MemoryAvailable gauge",
            r'MemoryAvailable\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE MemoryUtilization gauge",
            r'MemoryUtilization\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE DiskUsage gauge",
            r'DiskUsage\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE DiskUtilization gauge",
            r'DiskUtilization\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE DiskAvailable gauge",
            r'DiskAvailable\{Level="Host",Hostname=".+",\} \d+\.\d+',
            r"TYPE HandlerTime gauge",
            r'HandlerTime\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE PredictionTime gauge",
            r'PredictionTime\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE InferenceRequestCount counter",
            r'InferenceRequestCount\{Hostname=".+",\} \d+\.\d+',
            r"TYPE InitializeCallCount counter",
            r'InitializeCallCount\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE PreprocessCallCount counter",
            r'PreprocessCallCount\{ModelName="mnist_custom_metrics",Hostname=".+",\} \d+\.\d+',
            r"TYPE PostprocessCallCount counter",
            r'PostprocessCallCount\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE RequestBatchSize gauge",
            r'RequestBatchSize\{ModelName="mnist_custom_metrics",Hostname=".+",\} \d+\.\d+',
            r"TYPE SizeOfImage gauge",
            r'SizeOfImage\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE HandlerMethodTime gauge",
            r'HandlerMethodTime\{MethodName="preprocess",ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE ExamplePercentMetric histogram",
            r'ExamplePercentMetric_sum\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r'ExamplePercentMetric_count\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
        ]

        for pattern in prometheus_metric_patterns:
            matches = re.findall(pattern, prometheus_metrics)
            assert len(matches) == 1, "pattern not found " + pattern

    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_MODE"]
        del os.environ["TS_METRICS_CONFIG"]
        os.remove(config_file)


def test_auto_detect_backend_metrics_log_mode():
    """
    Validates that auto-detection of backend metrics works with log mode
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT,
        "test",
        "pytest",
        "test_data",
        "metrics",
        "metrics_auto_detect.yaml",
    )
    os.environ["TS_MODEL_METRICS_AUTO_DETECT"] = "true"

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("model_metrics.log", BACKEND_METRICS, True)
        validate_metrics_log("model_metrics.log", AUTO_DETECT_BACKEND_METRICS, True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_CONFIG"]
        del os.environ["TS_MODEL_METRICS_AUTO_DETECT"]
        os.remove(config_file)


def test_auto_detect_backend_metrics_prometheus_mode():
    """
    Validates that auto-detection of backend metrics works with prometheus mode
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_METRICS_MODE"] = "prometheus"
    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT,
        "test",
        "pytest",
        "test_data",
        "metrics",
        "metrics_auto_detect.yaml",
    )
    os.environ["TS_MODEL_METRICS_AUTO_DETECT"] = "true"

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()

        validate_metrics_log("model_metrics.log", BACKEND_METRICS, False)
        validate_metrics_log("model_metrics.log", AUTO_DETECT_BACKEND_METRICS, False)

        response = requests.get("http://localhost:8082/metrics")
        prometheus_metrics = response.text
        for metric_name in BACKEND_METRICS:
            assert metric_name in prometheus_metrics
        for metric_name in AUTO_DETECT_BACKEND_METRICS:
            assert metric_name in prometheus_metrics

        prometheus_metric_patterns = [
            r"TYPE HandlerTime gauge",
            r'HandlerTime\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE PredictionTime gauge",
            r'PredictionTime\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE InferenceRequestCount counter",
            r'InferenceRequestCount\{Hostname=".+",\} \d+\.\d+',
            r"TYPE InitializeCallCount counter",
            r'InitializeCallCount\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE PreprocessCallCount counter",
            r'PreprocessCallCount\{ModelName="mnist_custom_metrics",Hostname=".+",\} \d+\.\d+',
            r"TYPE PostprocessCallCount counter",
            r'PostprocessCallCount\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE RequestBatchSize gauge",
            r'RequestBatchSize\{ModelName="mnist_custom_metrics",Hostname=".+",\} \d+\.\d+',
            r"TYPE SizeOfImage gauge",
            r'SizeOfImage\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE HandlerMethodTime gauge",
            r'HandlerMethodTime\{MethodName="preprocess",ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r"TYPE ExamplePercentMetric histogram",
            r'ExamplePercentMetric_sum\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
            r'ExamplePercentMetric_count\{ModelName="mnist_custom_metrics",Level="Model",Hostname=".+",\} \d+\.\d+',
        ]

        for pattern in prometheus_metric_patterns:
            matches = re.findall(pattern, prometheus_metrics)
            assert len(matches) == 1, "pattern not found: " + pattern

    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_MODE"]
        del os.environ["TS_METRICS_CONFIG"]
        del os.environ["TS_MODEL_METRICS_AUTO_DETECT"]
        os.remove(config_file)


def test_collect_system_metrics_when_not_disabled():
    """
    Validates that system metrics are collected when not disabled
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT, "examples", "custom_metrics", "metrics.yaml"
    )

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("ts_metrics.log", SYSTEM_METRICS, True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_CONFIG"]
        os.remove(config_file)


def test_disable_system_metrics_using_config_properties():
    """
    Validates that system metrics collection is disabled when "disable_system_metrics"
    configuration option is set to "true"
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.writelines(["enable_envvars_config=true\n", "disable_system_metrics=true"])

    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT, "examples", "custom_metrics", "metrics.yaml"
    )

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("ts_metrics.log", SYSTEM_METRICS, False)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_METRICS_CONFIG"]
        os.remove(config_file)


def test_disable_system_metrics_using_environment_variable():
    """
    Validates that system metrics collection is disabled when TS_DISABLE_SYSTEM_METRICS
    environment variable is set to "true"
    """
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    # Remove existing logs if any
    for f in glob.glob("logs/*.log"):
        os.remove(f)

    config_file = os.path.join(test_utils.ROOT_DIR, "config.properties")
    with open(config_file, "w") as f:
        f.write("enable_envvars_config=true")

    os.environ["TS_DISABLE_SYSTEM_METRICS"] = "true"
    os.environ["TS_METRICS_CONFIG"] = os.path.join(
        test_utils.REPO_ROOT, "examples", "custom_metrics", "metrics.yaml"
    )

    try:
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=config_file,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request()
        validate_metrics_log("ts_metrics.log", SYSTEM_METRICS, False)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
        del os.environ["TS_DISABLE_SYSTEM_METRICS"]
        del os.environ["TS_METRICS_CONFIG"]
        os.remove(config_file)
