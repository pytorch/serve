import importlib
import os
import shutil
import sys

import pytest
import test_utils

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))

# Exclude the following tests from regression tests
collect_ignore = []
collect_ignore.append("test_example_torchrec_dlrm.py")
collect_ignore.append("test_example_near_real_time_video.py")
collect_ignore.append("test_dali_preprocess.py")
collect_ignore.append("test_example_intel_extension_for_pytorch.py")
collect_ignore.append("test_example_micro_batching.py")
# collect_ignore.append("test_example_scriptable_tokenzier.py")
collect_ignore.append("test_example_stateful_http.py")
# collect_ignore.append("test_gRPC_inference_api.py")
# collect_ignore.append("test_handler.py")
collect_ignore.append("test_metrics.py")
# collect_ignore.append("test_snapshot.py")
# collect_ignore.append("test_token_authorization.py")
# collect_ignore.append("test_pytorch_profiler.py")
collect_ignore.append("test_onnx.py")
# collect_ignore.append("test_model_archiver.py")
collect_ignore.append("test_metrics_kf.py")
# collect_ignore.append("test_model_custom_dependencies.py")
# collect_ignore.append("test_model_config.py")
# collect_ignore.append("test_parallelism.py")
# collect_ignore.append("test_gRPC_management_apis.py")
# collect_ignore.append("test_mnist_template.py")
# collect_ignore.append("test_ipex_serving.py")
# collect_ignore.append("test_handler_traceback_logging.py")
# collect_ignore.append("test_send_intermediate_prediction_response.py")


@pytest.fixture(scope="module")
def model_archiver():
    loader = importlib.machinery.SourceFileLoader(
        "archiver",
        os.path.join(
            REPO_ROOT_DIR, "model-archiver", "model_archiver", "model_packaging.py"
        ),
    )
    spec = importlib.util.spec_from_loader("archiver", loader)
    archiver = importlib.util.module_from_spec(spec)

    sys.modules["archiver"] = archiver

    loader.exec_module(archiver)

    yield archiver

    del sys.modules["archiver"]


@pytest.fixture(scope="module")
def model_store(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("work_dir")
    model_store_path = os.path.join(work_dir, "model_store")
    os.makedirs(model_store_path, exist_ok=True)

    yield model_store_path

    try:
        shutil.rmtree(model_store_path)
    except OSError:
        pass


@pytest.fixture(scope="module")
def torchserve(model_store):
    test_utils.torchserve_cleanup()

    pipe = test_utils.start_torchserve(
        model_store=model_store, no_config_snapshots=True, gen_mar=False
    )

    yield pipe

    test_utils.torchserve_cleanup()


@pytest.fixture(scope="session")
def monkeysession(request):
    """
    This fixture lets us create monkey patches in session scope like altering the Python path.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()
