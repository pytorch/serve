from ts_scripts.modelarchiver_utils import test_modelarchiver
from ts_scripts.backend_utils import test_torchserve
from ts_scripts.install_from_src import install_from_src
from ts_scripts.sanity_utils import test_sanity
from ts_scripts.shell_utils import rm_dir, rm_file
from ts_scripts.frontend_utils import test_frontend
import ts_scripts.tsutils as ts
import ts_scripts.print_env_info as build_hdr_printer


def torchserve_sanity():
    try:
        # Test frontend gradle
        test_frontend()

        # Install from src
        install_from_src()

        # Test Torchserve pylint, pytest
        test_torchserve()

        # Test Model archiver pylint, pytest, IT
        test_modelarchiver()

        # Run Sanity Tests
        test_sanity()
    finally:
        cleanup()


def cleanup():
    ts.stop_torchserve()
    rm_dir('model_store')
    rm_dir('logs')

    # clean up residual from model-archiver IT suite.
    rm_dir('model-archiver/model_archiver/htmlcov_ut model_archiver/model-archiver/htmlcov_it')
    rm_file('ts_scripts/*_pb2*.py', True)


if __name__ == '__main__':
    from pygit2 import Repository
    git_branch = Repository('.').head.shorthand
    build_hdr_printer.main(git_branch)

    torchserve_sanity()
