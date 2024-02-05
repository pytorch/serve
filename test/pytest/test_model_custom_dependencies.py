import os
import subprocess

import requests
import test_utils


def setup_module(module):
    test_utils.torchserve_cleanup()
    # Clean out custom model dependencies in base python environment
    subprocess.run(
        [
            "pip",
            "uninstall",
            "-y",
            "-r",
            str(
                os.path.join(
                    test_utils.REPO_ROOT,
                    "test",
                    "pytest",
                    "test_data",
                    "custom_dependencies",
                    "requirements.txt",
                )
            ),
        ],
        check=True,
    )


def teardown_module(module):
    test_utils.torchserve_cleanup()
    # Restore custom model dependencies in base python environment
    subprocess.run(
        [
            "pip",
            "install",
            "-r",
            str(
                os.path.join(
                    test_utils.REPO_ROOT,
                    "test",
                    "pytest",
                    "test_data",
                    "custom_dependencies",
                    "requirements.txt",
                )
            ),
        ],
        check=True,
    )


def generate_model_archive(use_requirements=False, use_venv=False):
    model_archiver_cmd = test_utils.model_archiver_command_builder(
        model_name="mnist_custom_dependencies",
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
            test_utils.REPO_ROOT,
            "test",
            "pytest",
            "test_data",
            "custom_dependencies",
            "mnist_custom_dependencies_handler.py",
        ),
        requirements_file=os.path.join(
            test_utils.REPO_ROOT,
            "test",
            "pytest",
            "test_data",
            "custom_dependencies",
            "requirements.txt",
        )
        if use_requirements
        else None,
        config_file=os.path.join(
            test_utils.REPO_ROOT,
            "test",
            "pytest",
            "test_data",
            "custom_dependencies",
            "model_config.yaml",
        )
        if use_venv
        else None,
        export_path=test_utils.MODEL_STORE,
        force=True,
    )
    model_archiver_cmd = model_archiver_cmd.split(" ")
    subprocess.run(model_archiver_cmd, check=True)


def register_model_and_make_inference_request(expect_model_load_failure=False):
    try:
        resp = test_utils.register_model(
            "mnist_custom_dependencies", "mnist_custom_dependencies.mar"
        )
        resp.raise_for_status()
    except Exception as e:
        if expect_model_load_failure:
            return
        else:
            raise e

    if expect_model_load_failure:
        raise Exception("Expected model load failure but model load succeeded")

    data_file = os.path.join(
        test_utils.REPO_ROOT,
        "examples",
        "image_classifier",
        "mnist",
        "test_data",
        "0.png",
    )
    with open(data_file, "rb") as input_data:
        resp = requests.post(
            url=f"http://localhost:8080/predictions/mnist_custom_dependencies",
            data=input_data,
        )
        resp.raise_for_status()


def test_install_dependencies_to_target_directory_with_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=True, use_venv=False)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=os.path.join(
                test_utils.REPO_ROOT,
                "test",
                "pytest",
                "test_data",
                "custom_dependencies",
                "config.properties",
            ),
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=False)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_install_dependencies_to_target_directory_without_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=False, use_venv=False)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=os.path.join(
                test_utils.REPO_ROOT,
                "test",
                "pytest",
                "test_data",
                "custom_dependencies",
                "config.properties",
            ),
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_disable_install_dependencies_to_target_directory_with_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=True, use_venv=False)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=None,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_disable_install_dependencies_to_target_directory_without_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=False, use_venv=False)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=None,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_install_dependencies_to_venv_with_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=True, use_venv=True)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=os.path.join(
                test_utils.REPO_ROOT,
                "test",
                "pytest",
                "test_data",
                "custom_dependencies",
                "config.properties",
            ),
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=False)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_install_dependencies_to_venv_without_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=False, use_venv=True)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=os.path.join(
                test_utils.REPO_ROOT,
                "test",
                "pytest",
                "test_data",
                "custom_dependencies",
                "config.properties",
            ),
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_disable_install_dependencies_to_venv_with_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=True, use_venv=True)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=None,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()


def test_disable_install_dependencies_to_venv_without_requirements():
    # Torchserve cleanup
    test_utils.stop_torchserve()
    test_utils.delete_all_snapshots()

    try:
        generate_model_archive(use_requirements=False, use_venv=True)
        test_utils.start_torchserve(
            model_store=test_utils.MODEL_STORE,
            snapshot_file=None,
            no_config_snapshots=True,
            gen_mar=False,
        )
        register_model_and_make_inference_request(expect_model_load_failure=True)
    finally:
        test_utils.stop_torchserve()
        test_utils.delete_all_snapshots()
