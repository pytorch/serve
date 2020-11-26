from ts_scripts.test_modelarchiver import test_modelarchiver
from ts_scripts.test_torchserve import test_torchserve
from ts_scripts.install_from_src import install_from_src
from ts_scripts.test_sanity import test_sanity
from ts_scripts.shell_utils import rm_dir
from ts_scripts.test_frontend import test_frontend
import ts_scripts.tsutils as ts
import test.print_env_info as build_hdr_printer


def torchserve_sanity():
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
    cleanup()


def cleanup():
    ts.stop_torchserve()
    rm_dir('model_store')
    rm_dir('logs')

    # clean up residual from model-archiver IT suite.
    rm_dir('model-archiver/model_archiver/htmlcov_ut model_archiver/model-archiver/htmlcov_it')


if __name__ == '__main__':
    from pygit2 import Repository
    git_branch = Repository('.').head.shorthand
    build_hdr_printer.main(git_branch)

    torchserve_sanity()
