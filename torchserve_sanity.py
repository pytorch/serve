import ts_scripts.tsutils as ts
from ts_scripts import marsgen as mg
from ts_scripts.backend_utils import test_torchserve
from ts_scripts.frontend_utils import test_frontend
from ts_scripts.install_from_src import install_from_src
from ts_scripts.modelarchiver_utils import test_modelarchiver
from ts_scripts.sanity_utils import (
    test_markdown_files,
    test_sanity,
    test_workflow_sanity,
)
from ts_scripts.shell_utils import rm_dir, rm_file
from ts_scripts.workflow_archiver_utils import test_workflow_archiver


def torchserve_sanity():
    try:
        # Test frontend gradle
        test_frontend()

        # Install from src
        install_from_src()

        # Generate mar files
        mg.generate_mars()

        # Test Torchserve pylint, pytest
        test_torchserve()

        # Test Model archiver pylint, pytest, IT
        test_modelarchiver()

        # Test Workflow archiver pylint, pytest, IT
        test_workflow_archiver()

        # Run Sanity Tests
        test_sanity()

        # Run workflow sanity
        test_workflow_sanity()

        # Check for broken links
        test_markdown_files()

    finally:
        cleanup()


def cleanup():
    ts.stop_torchserve()
    rm_dir("model_store")
    rm_dir("logs")

    # clean up residual from model-archiver IT suite.
    rm_dir(
        "model-archiver/model_archiver/htmlcov_ut model_archiver/model-archiver/htmlcov_it"
    )
    rm_file("ts_scripts/*_pb2*.py", True)

    # delete mar_gen_dir
    mg.delete_model_store_gen_dir()


if __name__ == "__main__":
    torchserve_sanity()
