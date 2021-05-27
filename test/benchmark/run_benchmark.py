import argparse
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-instances",
        action="store",
        help="Supply a .yaml file with test_name, instance_id, and key_filename to re-use already-running instances",
    )
    parser.add_argument(
        "--do-not-terminate",
        action="store_true",
        default=False,
        help="Use with caution: does not terminate instances, instead saves the list to a file in order to re-use",
    )

    arguments = parser.parse_args()
    do_not_terminate_string = "" if not arguments.do_not_terminate else "--do-not-terminate"
    use_instances_arg_list = ["--use-instances", f"{arguments.use_instances}"] if arguments.use_instances else []

    # Run this script from the root directory 'serve', it changes directory below as required
    os.chdir(os.path.join(os.getcwd(), "test", "benchmark"))

    execution_id = f"ts-benchmark-run-{str(uuid.uuid4())}"

    test_path = os.path.join(os.getcwd(), "tests")
    LOGGER.info(f"Running tests from directory: {test_path}")

    pytest_args = [
        "-s",
        "-rA",
        test_path,
        "-n=4",
        "--disable-warnings",
        "-v",
        "--execution-id",
        execution_id,
        do_not_terminate_string,
    ] + use_instances_arg_list

    LOGGER.info(f"Running pytest")

    pytest.main(pytest_args)


if __name__ == "__main__":
    main()
