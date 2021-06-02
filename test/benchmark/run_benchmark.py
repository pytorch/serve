import os
import random
import sys
import logging
import re
import uuid

import boto3
import pytest

from invoke import run
from invoke.context import Context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def main():
    # Run this script from the root directory 'serve', it changes directory below as required
    os.chdir(os.path.join(os.getcwd(), "test", "benchmark"))

    execution_id = f"ts-benchmark-run-{str(uuid.uuid4())}"

    test_path = os.path.join(os.getcwd(), "tests")
    LOGGER.info(f"Running tests from directory: {test_path}")

    pytest_args = ["-s", "-rA", test_path, "-n=4", "--disable-warnings", "-v", "--execution-id", execution_id]

    LOGGER.info(f"Running pytest")

    pytest.main(pytest_args)


if __name__ == "__main__":
    main()
