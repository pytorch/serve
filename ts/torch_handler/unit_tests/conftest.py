import pytest

from .test_utils.mock_context import MockContext


@pytest.fixture()
def model_context():
    return MockContext()


# @pytest.fixture(autouse=True, scope="class")
# def setup_directories():

#     TEST_DIR = os.path.join(
#         "ts", "torch", "torch_handler", "unit_tests", "models", "tmp"
#     )
#     sys.path.append(TEST_DIR)

#     os.system(f"mkdir -p {TEST_DIR}")
#     yield
#     os.system(f"rm -rf {TEST_DIR}")


# Function for create, download or move model
