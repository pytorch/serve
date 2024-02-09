import glob
import json
import os
import platform
import time
from pathlib import Path

import requests
import test_utils


def setup_module(module):
    test_utils.torchserve_cleanup()


def teardown_module(module):
    test_utils.torchserve_cleanup()


def replace_mar_file_with_dummy_mar_in_model_store(model_store=None, model_mar=None):
    model_store = (
        model_store
        if (model_store != None)
        else os.path.join(test_utils.ROOT_DIR, "model_store")
    )
    if model_mar != None:
        myfilepath = os.path.join(model_store, model_mar)
        if os.path.exists(myfilepath):
            os.remove(myfilepath)
            with open(myfilepath, "w+") as f:
                f.write("junk data")


def test_snapshot_created_on_start_and_stop():
    """
    Validates that startup .cfg & shutdown.cfg are created upon start & stop.
    """
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve()
    test_utils.stop_torchserve()
    assert len(glob.glob("logs/config/*startup.cfg")) == 1
    if platform.system() != "Windows":
        assert len(glob.glob("logs/config/*shutdown.cfg")) == 1


def snapshot_created_on_management_api_invoke(model_mar="densenet161.mar"):
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve()
    mar_path = "mar_path_{}".format(model_mar[0:-4])
    if mar_path in test_utils.mar_file_table:
        requests.post("http://127.0.0.1:8081/models?url=" + model_mar)
    else:
        requests.post(
            "http://127.0.0.1:8081/models?url=https://torchserve.pytorch.org/mar_files/"
            + model_mar
        )
    time.sleep(10)
    test_utils.stop_torchserve()


def test_snapshot_created_on_management_api_invoke():
    """
    Validates that snapshot.cfg is created when management apis are invoked.
    """
    snapshot_created_on_management_api_invoke()
    assert len(glob.glob("logs/config/*snap*.cfg")) == 1


def test_start_from_snapshot():
    """
    Validates if we can restore state from snapshot.
    """
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[0]
    test_utils.start_torchserve(snapshot_file=snapshot_cfg)
    response = requests.get("http://127.0.0.1:8081/models/")
    assert json.loads(response.content)["models"][0]["modelName"] == "densenet161"
    test_utils.stop_torchserve()


def test_start_from_latest():
    """
    Validates if latest snapshot file is picked if we dont pass snapshot arg explicitly.
    """
    test_utils.start_torchserve()
    response = requests.get("http://127.0.0.1:8081/models/")
    assert json.loads(response.content)["models"][0]["modelName"] == "densenet161"
    test_utils.stop_torchserve()


def test_start_from_read_only_snapshot():
    """
    Validates if we can start and restore Torchserve state using a read-only snapshot.
    """
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[0]
    file_status = os.stat(snapshot_cfg)
    os.chmod(snapshot_cfg, 0o444)
    test_utils.start_torchserve(snapshot_file=snapshot_cfg)
    os.chmod(snapshot_cfg, (file_status.st_mode & 0o777))
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
    except:
        assert (
            False
        ), "Something is not right!! Failed to start Torchserve using Read Only Snapshot!!"
    else:
        assert (
            True
        ), "Successfully started and restored Torchserve state using a Read Only Snapshot"


def test_no_config_snapshots_cli_option():
    """
    Validates that --no-config-snapshots works as expected.
    """
    # Required to stop torchserve here so that all config files gets deleted
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve(no_config_snapshots=True)
    test_utils.stop_torchserve()
    assert len(glob.glob("logs/config/*.cfg")) == 0


def test_start_from_default():
    """
    Validates that Default config is used if we dont use a config explicitly.
    """
    test_utils.delete_all_snapshots()
    test_utils.start_torchserve()
    response = requests.get("http://127.0.0.1:8081/models/")
    assert len(json.loads(response.content)["models"]) == 0


def test_start_from_non_existing_snapshot():
    """
    Validates that Torchserve should fail to start when we pass a non-existent snapshot
     as an input snapshot while starting Torchserve.
    """
    test_utils.stop_torchserve()
    test_utils.start_torchserve(snapshot_file="logs/config/junk-snapshot.cfg")
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
    except:
        assert True, "Failed to start Torchserve using a Non Existing Snapshot"
    else:
        assert False, (
            "Something is not right!! Successfully started Torchserve "
            "using Non Existing Snapshot File!!"
        )


def test_torchserve_init_with_non_existent_model_store():
    """Validates that Torchserve fails to start if the model store directory is non existent"""

    test_utils.start_torchserve(
        model_store="/invalid_model_store", snapshot_file=None, no_config_snapshots=True
    )
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
    except:
        assert (
            True
        ), "Failed to start Torchserve using non existent model-store directory"
    else:
        assert False, (
            "Something is not right!! Successfully started Torchserve "
            "using non existent directory!!"
        )
    finally:
        test_utils.delete_model_store()
        test_utils.delete_all_snapshots()


def test_restart_torchserve_with_last_snapshot_with_model_mar_removed():
    """Validates that torchserve will fail to start in the following scenario:
    1) We use a snapshot file to start torchserve. The snapshot contains reference to "A" model file
    2) The "A" model mar file is accidentally deleted from the model store"""

    # Register model using mgmt api
    snapshot_created_on_management_api_invoke()

    # Now remove the registered model mar file (delete_mar_ fn)
    test_utils.delete_mar_file_from_model_store(
        model_store=os.path.join(test_utils.ROOT_DIR, "model_store"),
        model_mar="densenet",
    )

    # Start Torchserve with last generated snapshot file
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[0]
    test_utils.start_torchserve(snapshot_file=snapshot_cfg, gen_mar=False)
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
    except:
        assert (
            True
        ), "Failed to start Torchserve properly as reqd model mar file is missing!!"
    else:
        assert (
            False
        ), "Something is not right!! Successfully started Torchserve without reqd mar file"
    finally:
        test_utils.delete_model_store()
        test_utils.delete_all_snapshots()


def test_replace_mar_file_with_dummy():
    """Validates that torchserve will fail to start in the following scenario:
    1) We use a snapshot file to start torchserve. The snapshot contains reference to "A" model file
    2) "A" model file gets corrupted or is replaced by some dummy mar file with same name
    """

    snapshot_created_on_management_api_invoke()

    # Start Torchserve using last snapshot state
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[0]
    test_utils.start_torchserve(snapshot_file=snapshot_cfg)
    response = requests.get("http://127.0.0.1:8081/models/")
    assert json.loads(response.content)["models"][0]["modelName"] == "densenet161"
    test_utils.stop_torchserve()

    # Now replace the registered model mar with dummy file
    replace_mar_file_with_dummy_mar_in_model_store(
        model_store=os.path.join(test_utils.ROOT_DIR, "model_store"),
        model_mar="densenet161.mar",
    )
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[0]
    test_utils.start_torchserve(snapshot_file=snapshot_cfg, gen_mar=False)
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
        assert json.loads(response.content)["models"][0]["modelName"] == "densenet161"
    except:
        assert False, "Default manifest does not work"
    else:
        assert (
            True
        ), "Successfully started Torchserve with a dummy mar file (ie. default manifest)"
    finally:
        test_utils.unregister_model("densenet161")
        test_utils.delete_all_snapshots()
        test_utils.delete_model_store()
        test_utils.stop_torchserve()


def test_restart_torchserve_with_one_of_model_mar_removed():
    """Validates that torchserve will fail to start in the following scenario:
    1) We use a snapshot file to start torchserve. The snapshot contains reference to few model files
    2) One of these model mar files are accidentally deleted from the model store"""
    # Register multiple models
    # 1st model
    test_utils.delete_model_store()
    test_utils.start_torchserve()
    requests.post("http://127.0.0.1:8081/models?url=densenet161.mar")
    time.sleep(15)
    # 2nd model
    requests.post("http://127.0.0.1:8081/models?url=mnist.mar")
    time.sleep(15)
    test_utils.stop_torchserve()

    # Start Torchserve
    test_utils.start_torchserve()
    response = requests.get("http://127.0.0.1:8081/models/")
    num_of_regd_models = len(json.loads(response.content)["models"])
    test_utils.stop_torchserve()

    # Now remove the registered model mar file (delete_mar_ fn)
    test_utils.delete_mar_file_from_model_store(
        model_store=os.path.join(test_utils.ROOT_DIR, "model_store"),
        model_mar="densenet",
    )

    # Start Torchserve with existing snapshot file containing reference to one of the model mar file
    # which is now missing from the model store
    snapshot_cfg = glob.glob("logs/config/*snap*.cfg")[1]
    test_utils.start_torchserve(snapshot_file=snapshot_cfg, gen_mar=False)
    try:
        response = requests.get("http://127.0.0.1:8081/models/")
    except:
        assert (
            True
        ), "Failed to start Torchserve as one of reqd model mar file is missing"
    else:
        assert False, (
            "Something is not right!! Started Torchserve successfully with a "
            "reqd model mar file missing from the model store!!"
        )
    finally:
        test_utils.torchserve_cleanup()


def test_empty_runtime():
    test_utils.delete_all_snapshots()
    test_utils.stop_torchserve()
    test_utils.start_torchserve()
    requests.post("http://127.0.0.1:8081/models?url=mnist.mar")
    test_utils.stop_torchserve()

    cfgs = glob.glob("logs/config/*shutdown.cfg")
    assert len(cfgs) == 1

    def remove_runtime_type(json_str):
        # Remove the prefix 'model_snapshot=' from the input string
        model_snapshot = json.loads(
            json_str[len("model_snapshot=") :].replace("\\:", ":").replace("\\n", "")
        )

        # Remove the "runtimeType" element from the JSON object
        for model in model_snapshot["models"].values():
            for version, config in model.items():
                del config["runtimeType"]

        # Return the modified JSON object as a string with the original prefix
        return "model_snapshot=" + json.dumps(model_snapshot, indent=2).replace(
            "\n", "\\n"
        ).replace(":", "\\:")

    cfg_text = Path(cfgs[0]).read_text().split("\n")
    model_snapshot = [line for line in cfg_text if line.startswith("model_snapshot")][0]
    cfg_text = [line for line in cfg_text if not line.startswith("model_snapshot")]
    cfg_text += [remove_runtime_type(model_snapshot)]
    Path(cfgs[0]).write_text("\n".join(cfg_text))

    test_utils.start_torchserve()

    try:
        requests.get("http://127.0.0.1:8081/models/")
    except:
        assert False, "Could not start TorchServe."
