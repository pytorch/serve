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
collect_ignore.append("test_example_scriptable_tokenzier.py")


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
