import datetime
import logging
import os
import random
import re
import sys

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from fabric2 import Connection
from invoke import run
from invoke.context import Context


import tests.utils.ec2 as ec2_utils
import tests.utils.s3 as s3_utils
from tests.utils import (
    AMI_ID,
    DEFAULT_REGION,
    IAM_INSTANCE_PROFILE,
    LOGGER,
    DockerImageHandler,
    YamlHandler,
    DEFAULT_DOCKER_DEV_ECR_REPO,
    S3_BUCKET_BENCHMARK_ARTIFACTS,
)
from tests.utils.s3 import ArtifactsHandler


def pytest_addoption(parser):
    parser.addoption(
        "--execution-id",
        default="123456789",
        action="store",
        help="execution id that is used to keep all artifacts together",
    )


@pytest.fixture(scope="session", autouse=True)
def build_docker_container(request, docker_dev_image_config_path):
    LOGGER.info(f"Setting up docker image to be used")
    docker_config = YamlHandler.load_yaml(docker_dev_image_config_path)
    YamlHandler.validate_docker_yaml(docker_config)

    account_id = run("aws sts get-caller-identity --query Account --output text").stdout.strip()

    for processor, config in docker_config.items():
        docker_tag = None
        cuda_version = None
        for config_key, config_value in config.items():
            if processor == "gpu" and config_key == "cuda_version":
                cuda_version = config_value
            if config_key == "docker_tag":
                docker_tag = config_value
        dockerImageHandler = DockerImageHandler(docker_tag, cuda_version)
        dockerImageHandler.build_image()
        dockerImageHandler.push_docker_image_to_ecr(
            account_id, DEFAULT_REGION, f"{DEFAULT_DOCKER_DEV_ECR_REPO}:{docker_tag}"
        )


@pytest.fixture(scope="session")
def docker_dev_image_config_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "docker", "docker.yaml")


@pytest.fixture(scope="session", autouse=True)
def benchmark_execution_id(request):
    execution_id = request.config.getoption("--execution-id")
    LOGGER.info(
        f"execution id for this run is : {execution_id}. Server logs for each benchmark run are available at S3 location: {S3_BUCKET_BENCHMARK_ARTIFACTS}/{execution_id}"
    )
    return execution_id


@pytest.fixture(scope="function")
def vgg11_config_file_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "vgg11.yaml")


@pytest.fixture(scope="function")
def vgg16_config_file_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "vgg16.yaml")


@pytest.fixture(scope="function")
def bert_config_file_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "bert.yaml")


@pytest.fixture(scope="function")
def mnist_config_file_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "mnist.yaml")


@pytest.fixture(scope="function")
def fastrcnn_config_file_path(request):
    return os.path.join(os.getcwd(), "tests", "suite", "fastrcnn.yaml")


@pytest.fixture(scope="session")
def region():
    return os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION)


@pytest.fixture(scope="session")
def docker_client(region):
    test_utils.run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {region})",
        failure="Failed to log into ECR.",
    )
    return docker.from_env()


@pytest.fixture(scope="session")
def ecr_client(region):
    return boto3.client("ecr", region_name=region)


@pytest.fixture(scope="function")
def ec2_key_name(request):
    random.seed(datetime.datetime.now())
    return f"{request.fixturename.replace('_', '-')}-{random.randrange(1, 10000, 3)}"


@pytest.fixture(scope="session")
def ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


@pytest.fixture(scope="session")
def ec2_resource(region):
    return boto3.resource("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


@pytest.fixture(scope="function")
def ec2_instance_type(request):
    return request.param if hasattr(request, "param") else "c4.4xlarge"


@pytest.fixture(scope="function")
def ec2_instance_role_name(request):
    return request.param if hasattr(request, "param") else IAM_INSTANCE_PROFILE


@pytest.fixture(scope="function")
def ec2_instance_ami(request):
    return request.param if hasattr(request, "param") else AMI_ID


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ec2_instance(
    request,
    ec2_client,
    ec2_resource,
    ec2_instance_type,
    ec2_key_name,
    ec2_instance_role_name,
    ec2_instance_ami,
    region,
):
    key_filename = ec2_utils.generate_ssh_keypair(ec2_client, ec2_key_name)

    def delete_ssh_keypair():
        ec2_utils.destroy_ssh_keypair(ec2_client, key_filename)

    request.addfinalizer(delete_ssh_keypair)

    params = {
        "KeyName": ec2_key_name,
        "ImageId": ec2_instance_ami,
        "InstanceType": ec2_instance_type,
        "IamInstanceProfile": {"Name": ec2_instance_role_name},
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"TS Benchmark {ec2_key_name}"}]},
        ],
        "MaxCount": 1,
        "MinCount": 1,
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 120}}],
    }

    try:
        instances = ec2_resource.create_instances(**params)
    except ClientError as e:
        if e.response["Error"]["Code"] == "InsufficientInstanceCapacity":
            LOGGER.warning(f"Failed to launch {ec2_instance_type} in {region} because of insufficient capacity")
        raise
    instance_id = instances[0].id

    LOGGER.info(f"Created instance: TS Benchmark {ec2_key_name}")

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(instance_id, system_status="ok", instance_status="ok", region=region)
    return instance_id, key_filename


@pytest.fixture(scope="function")
def ec2_connection(request, ec2_instance, ec2_instance_type, region):
    """
    Fixture to establish connection with EC2 instance if necessary
    :param request: pytest test request
    :param ec2_instance: ec2_instance pytest fixture
    :param ec2_instance_type: ec2_instance_type pytest fixture
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    instance_id, instance_pem_file = ec2_instance
    ip_address = ec2_utils.get_public_ip(instance_id, region=region)
    LOGGER.info(f"Instance ip_address: {ip_address}")
    user = ec2_utils.get_instance_user(instance_id, region=region)
    LOGGER.info(f"Connecting to {user}@{ip_address}")
    conn = Connection(user=user, host=ip_address, connect_kwargs={"key_filename": [instance_pem_file]})

    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 100000)

    artifact_folder = f"ts-benchmark-{unique_id}-folder"

    ArtifactsHandler.upload_torchserve_folder_to_instance(conn, artifact_folder)

    def delete_s3_artifact_copy():
        ArtifactsHandler.cleanup_temp_s3_folder(artifact_folder)

    request.addfinalizer(delete_s3_artifact_copy)


    return conn

