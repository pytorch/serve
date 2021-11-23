import argparse
import boto3
import datetime
import random
import subprocess
import os
import time


from botocore.config import Config
from fabric2 import Connection
from invoke import run

from utils import LOGGER, GPU_INSTANCES
from utils import ec2 as ec2_utils

CPU_INSTANCE_COMMANDS_LIST = [
    "python3 ts_scripts/install_dependencies.py --environment=dev",
    "python3 torchserve_sanity.py",
    "cd serving-sdk/ && mvn clean install -q && cd ../",
]

GPU_INSTANCE_COMMANDS_LIST = [
    "python3 ts_scripts/install_dependencies.py --environment=dev --cuda=cu102",
    "python3 torchserve_sanity.py",
    "cd serving-sdk/ && mvn clean install -q && cd ../",
]


def run_commands_on_ec2_instance(ec2_connection, is_gpu):
    """
    This function assumes that the required 'serve' folder is already available on the ec2 instance in the home directory.
    Returns a map of the command executed and return value of that command.
    """

    command_result_map = {}

    virtual_env_name = "venv"

    with ec2_connection.cd(f"/home/ubuntu/serve"):
        ec2_connection.run(f"python3 -m venv {virtual_env_name}")
        with ec2_connection.prefix(f"source {virtual_env_name}/bin/activate"):
            commands_list = GPU_INSTANCE_COMMANDS_LIST if is_gpu else CPU_INSTANCE_COMMANDS_LIST

            for command in commands_list:
                LOGGER.info(f"*** Executing command on ec2 instance: {command}")
                ret_obj = ec2_connection.run(
                    command,
                    echo=True,
                    warn=True,
                    shell="/bin/bash",
                    env={
                        "LC_CTYPE": "en_US.utf8",
                        "JAVA_HOME": "/usr/lib/jvm/java-11-openjdk-amd64",
                        "PYTHONIOENCODING": "utf8",
                    },
                    encoding="utf8"
                )

                if ret_obj.return_code != 0:
                    LOGGER.error(f"*** Failed command: {command}")
                    LOGGER.error(f"*** Failed command stdout: {ret_obj.stdout}")
                    LOGGER.error(f"*** Failed command stderr: {ret_obj.stderr}")

                command_result_map[command] = ret_obj.return_code

    return command_result_map


def launch_ec2_instance(region, instance_type, ami_id):
    """
    Note: This function relies on CODEBUILD environment variables. If this function is used outside of CODEBUILD,
    modify the function accordingly.
    Spins up an ec2 instance, clones the current Github Pull Request commit id on the instance, and runs sanity test on it.
    Prints the output of the command executed.
    """
    github_repo = os.environ.get("CODEBUILD_SOURCE_REPO_URL", "https://github.com/pytorch/serve.git").strip()
    github_pr_commit_id = os.environ.get("CODEBUILD_RESOLVED_SOURCE_VERSION", "HEAD").strip()
    github_hookshot = os.environ.get("CODEBUILD_SOURCE_VERSION", "job-local").strip()
    github_hookshot = github_hookshot.replace("/", "-")

    # Extract the PR number or use the last 6 characters of the commit id
    github_pull_request_number = github_hookshot.split("-")[1] if "-" in github_hookshot else github_hookshot[-6:]

    ec2_client = boto3.client("ec2", config=Config(retries={"max_attempts": 10}), region_name=region)
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    ec2_key_name = f"{github_hookshot}-ec2-instance-{random.randint(1, 1000)}"

    # Spin up ec2 instance and run tests
    try:
        key_file = ec2_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
        instance_details = ec2_utils.launch_instance(
            ami_id,
            instance_type,
            ec2_key_name=ec2_key_name,
            region=region,
            user_data=None,
            iam_instance_profile_name=ec2_utils.EC2_INSTANCE_ROLE_NAME,
            instance_name=ec2_key_name,
        )

        instance_id = instance_details["InstanceId"]
        ip_address = ec2_utils.get_public_ip(instance_id, region=region)

        LOGGER.info(f"*** Waiting on instance checks to complete...")
        ec2_utils.check_instance_state(instance_id, state="running", region=region)
        ec2_utils.check_system_state(instance_id, system_status="ok", instance_status="ok", region=region)
        LOGGER.info(f"*** Instance checks complete. Running commands on instance.")

        # Create a fabric connection to the ec2 instance.
        ec2_connection = ec2_utils.get_ec2_fabric_connection(instance_id, key_file, region)

        LOGGER.info(f"Running update command. This could take a while.")
        ec2_connection.run(f"sudo apt update")

        # Update command takes a while to run, and should ideally run uninterrupted
        time.sleep(300)

        with ec2_connection.cd("/home/ubuntu"):
            LOGGER.info(f"*** Cloning the PR related to {github_hookshot} on the ec2 instance.")
            ec2_connection.run(f"git clone {github_repo}")
            ec2_connection.run(
                f"cd serve && git fetch origin pull/{github_pull_request_number}/head:pull && git checkout pull"
            )

            ec2_connection.run(f"sudo apt-get install -y python3-venv")
            # Following is necessary on Base Ubuntu DLAMI because the default python is python2
            # This will NOT fail for other AMI where default python is python3
            ec2_connection.run(
                f"sudo cp /usr/local/bin/pip3 /usr/local/bin/pip && pip install --upgrade pip", warn=True
            )
            ec2_connection.run(
                f"sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1", warn=True
            )

        is_gpu = True if instance_type[:2] in GPU_INSTANCES else False

        command_return_value_map = run_commands_on_ec2_instance(ec2_connection, is_gpu)

        if any(command_return_value_map.values()):
            raise ValueError(f"*** One of the commands executed on ec2 returned a non-zero value.")
        else:
            LOGGER.info(f"*** All commands executed successfully on ec2. command:return_value map is as follows:")
            LOGGER.info(command_return_value_map)

    except ValueError as e:
        LOGGER.error(f"*** ValueError: {e}")
        LOGGER.error(f"*** Following commands had the corresponding return value:")
        LOGGER.error(command_return_value_map)
        raise e
    except Exception as e:
        LOGGER.error(f"*** Exception occured. {e}")
        raise e
    finally:
        LOGGER.warning(f"*** Terminating instance-id: {instance_id} with name: {ec2_key_name}")
        ec2_utils.terminate_instance(instance_id, region)
        LOGGER.warning(f"*** Destroying ssh key_pair: {ec2_key_name}")
        ec2_utils.destroy_ssh_keypair(ec2_client, ec2_key_name)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instance-type",
        default="p3.2xlarge",
        help="Specify the instance type you want to run the test on. Default: p3.2xlarge",
    )

    parser.add_argument(
        "--region",
        default="us-west-2",
        help="Specify the aws region in which you want associated ec2 instance to be spawned",
    )

    parser.add_argument(
        "--ami-id",
        default="ami-032e40ca6b0973cf2",
        help="Specify an Ubuntu Base DLAMI only. This AMI type ships with nvidia drivers already setup. Using other AMIs might"
        "need non-trivial installations on the AMI. AMI-ids differ per aws region.",
    )

    arguments = parser.parse_args()

    instance_type = arguments.instance_type
    region = arguments.region
    ami_id = arguments.ami_id

    launch_ec2_instance(region, instance_type, ami_id)


if __name__ == "__main__":
    main()
