import json
from pathlib import Path

import pytest
import test_utils

REPO_ROOT = Path(__file__).parents[3]
SANITY_MODELS_CONFIG = REPO_ROOT.joinpath("ts_scripts", "configs", "sanity_models.json")


def load_resnet18() -> dict:
    with open(SANITY_MODELS_CONFIG) as f:
        models = json.load(f)
    return list(filter(lambda x: x["name"] == "resnet-18", models))[0]


@pytest.fixture(name="resnet18")
def generate_resnet18(model_store, gen_models, ts_scripts_path):
    model = load_resnet18()

    from ts_scripts.marsgen import generate_model

    generate_model(gen_models[model["name"]], model_store)

    yield model


@pytest.fixture(scope="module")
def torchserve_with_snapshot(model_store):
    test_utils.torchserve_cleanup()

    test_utils.start_torchserve(
        model_store=model_store, no_config_snapshots=False, gen_mar=False
    )

    yield

    test_utils.torchserve_cleanup()


def test_config_snapshotting(
    resnet18, model_store, torchserve_with_snapshot, ts_scripts_path
):
    from ts_scripts.sanity_utils import run_rest_test

    run_rest_test(resnet18, unregister_model=False)

    test_utils.stop_torchserve()

    test_utils.start_torchserve(
        model_store=model_store, no_config_snapshots=False, gen_mar=False
    )

    run_rest_test(resnet18, register_model=False)
