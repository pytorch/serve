from scripts.test_modelarchiver import test_modelarchiver
from scripts.test_torchserve import test_torchserve
from scripts.install_from_src import install_from_src
from scripts.test_sanity import test_sanity


def torchserve_sanity():
    # Install from src
    install_from_src()

    # Test Torchserve pylint, pytest
    test_torchserve()

    # Test Model archiver pylint, pytest, IT
    test_modelarchiver()

    # Run Sanity Tests
    test_sanity()
    # cleanup()


# def cleanup():
#     stop_torchserve
#
#     rm - rf model_store
#     rm - rf logs
#
#     # clean up residual from model-archiver IT suite.
#     rm - rf model_archiver / model - archiver / htmlcov_ut
#     model_archiver / model - archiver / htmlcov_it
#     }

if __name__ == '__main__':
    torchserve_sanity()
