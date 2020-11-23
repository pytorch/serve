from scripts.test_modelarchiver import test_modelarchiver
from scripts.test_torchserve import test_torchserve
from scripts.install_from_src import install_from_src
from scripts.test_sanity import test_sanity
from scripts.shell_utils import rm_dir, rm_file
from scripts.test_frontend import test_frontend
import scripts.tsutils as ts
import test.print_env_info as build_hdr_printer


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
    rm_file('scripts/*_pb2*.py', True)


if __name__ == '__main__':
    from pygit2 import Repository
    git_branch = Repository('.').head.shorthand
    build_hdr_printer.main(git_branch)

    torchserve_sanity()
