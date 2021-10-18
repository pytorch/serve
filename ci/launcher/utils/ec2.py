import boto3
import os
import time
import re
from inspect import signature
import random

from retrying import retry
from fabric2 import Connection
from botocore.config import Config
from botocore.exceptions import ClientError
from invoke import run
from invoke.context import Context

from . import DEFAULT_REGION, LOGGER 

EC2_INSTANCE_ROLE_NAME = "ec2InstanceCIRole"


def generate_ssh_keypair(ec2_client, key_name):
    pwd = run("pwd", hide=True).stdout.strip("\n")
    key_filename = os.path.join(pwd, f"{key_name}.pem")
    if os.path.exists(key_filename):
        run(f"chmod 400 {key_filename}")
        return key_filename
    try:
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
    except ClientError as e:
        if "InvalidKeyPair.Duplicate" in f"{e}":
            # Wait 10 seconds for key to be created to avoid race condition
            time.sleep(10)
            if os.path.exists(key_filename):
                run(f"chmod 400 {key_filename}")
                return key_filename
        raise e

    run(f"echo '{key_pair['KeyMaterial']}' > {key_filename}")
    run(f"chmod 400 {key_filename}")
    return key_filename


def destroy_ssh_keypair(ec2_client, key_filename):
    key_name = os.path.basename(key_filename).split(".pem")[0]
    response = ec2_client.delete_key_pair(KeyName=key_name)
    run(f"rm -f {key_filename}")
    return response, key_name


def launch_instance(
    ami_id,
    instance_type,
    ec2_key_name=None,
    region="us-west-2",
    user_data=None,
    iam_instance_profile_name=None,
    instance_name="",
):
    """
    Launch an instance
    :param ami_id: AMI ID to be used for launched instance
    :param instance_type: Instance type of launched instance
    :param region: Region where instance will be launched
    :param user_data: Script to run when instance is launched as a str
    :param iam_instance_profile_arn: EC2 Role to be attached
    :param instance_name: Tag to display as Name on EC2 Console
    :return: <dict> Information about the instance that was launched
    """
    if not ami_id:
        raise Exception("No ami_id provided")
    if not ec2_key_name:
        raise Exception("Ec2 Key name must be provided")
    client = boto3.Session(region_name=region).client("ec2")

    # Construct the dictionary with the arguments for API call
    arguments_dict = {
        "KeyName": ec2_key_name,
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MaxCount": 1,
        "MinCount": 1,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"CI-CD {instance_name}"}],
            },
        ],
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 200,
                },
            }
        ],
    }

    if user_data:
        arguments_dict["UserData"] = user_data
    if iam_instance_profile_name:
        arguments_dict["IamInstanceProfile"] = {"Name": iam_instance_profile_name}

    LOGGER.info(f"Launching instance with name: {instance_name}, and key: {ec2_key_name}")

    response = client.run_instances(**arguments_dict)

    if not response or len(response["Instances"]) < 1:
        raise Exception("Unable to launch the instance. Did not return any response")
    
    LOGGER.info(f"Instance launched successfully.")

    return response["Instances"][0]


def get_ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


def get_instance_from_id(instance_id, region=DEFAULT_REGION):
    """
    Get instance information using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    instance = client.describe_instances(InstanceIds=[instance_id])
    if not instance:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    return instance["Reservations"][0]["Instances"][0]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_public_ip(instance_id, region=DEFAULT_REGION):
    """
    Get Public IP of instance using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> IP Address of instance with matching instance ID
    """
    instance = get_instance_from_id(instance_id, region)
    if not instance["PublicIpAddress"]:
        raise Exception("IP address not yet available")
    return instance["PublicIpAddress"]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_public_ip_from_private_dns(private_dns, region=DEFAULT_REGION):
    """
    Get Public IP of instance using private DNS
    :param private_dns:
    :param region:
    :return: <str> IP Address of instance with matching private DNS
    """
    client = boto3.Session(region_name=region).client("ec2")
    response = client.describe_instances(Filters={"Name": "private-dns-name", "Value": [private_dns]})
    return response.get("Reservations")[0].get("Instances")[0].get("PublicIpAddress")


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_instance_user(instance_id, region=DEFAULT_REGION):
    """
    Get "ubuntu" or "ec2-user" based on AMI used to launch instance
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> user name
    """
    instance = get_instance_from_id(instance_id, region)
    # Modify here if an AMI other than Ubuntu AMI must be used.
    user = "ubuntu"
    return user


def get_instance_state(instance_id, region=DEFAULT_REGION):
    """
    Get state of instance using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> State of instance with matching instance ID
    """
    instance = get_instance_from_id(instance_id, region)
    return instance["State"]["Name"]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def check_instance_state(instance_id, state="running", region=DEFAULT_REGION):
    """
    Compares the instance state with the state argument.
    Retries 8 times with 120 seconds gap between retries.
    :param instance_id: Instance ID to be queried
    :param state: Expected instance state
    :param region: Region where query will be performed
    :return: <str> State of instance with matching instance ID
    """
    instance_state = get_instance_state(instance_id, region)
    if state != instance_state:
        raise Exception(f"Instance {instance_id} not in {state} state")
    return instance_state


def get_system_state(instance_id, region=DEFAULT_REGION):
    """
    Returns health checks state for instances
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <tuple> System state and Instance state of instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    response = client.describe_instance_status(InstanceIds=[instance_id])
    if not response:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    instance_status_list = response["InstanceStatuses"]
    if not instance_status_list:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    if len(instance_status_list) < 1:
        raise Exception(
            "The instance id seems to be incorrect {}. \
                         reservations seems to be empty".format(
                instance_id
            )
        )

    instance_status = instance_status_list[0]
    return (
        instance_status["SystemStatus"]["Status"],
        instance_status["InstanceStatus"]["Status"],
    )


@retry(stop_max_attempt_number=96, wait_fixed=10000)
def check_system_state(instance_id, system_status="ok", instance_status="ok", region=DEFAULT_REGION):
    """
    Compares the system state (Health Checks).
    Retries 96 times with 10 seconds gap between retries
    :param instance_id: Instance ID to be queried
    :param system_status: Expected system state
    :param instance_status: Expected instance state
    :param region: Region where query will be performed
    :return: <tuple> System state and Instance state of instance with matching instance ID
    """
    instance_state = get_system_state(instance_id, region=region)
    if system_status != instance_state[0] or instance_status != instance_state[1]:
        raise Exception(
            "Instance {} not in \
                         required state".format(
                instance_id
            )
        )
    return instance_state


def terminate_instance(instance_id, region=DEFAULT_REGION):
    """
    Terminate EC2 instances with matching instance ID
    :param instance_id: Instance ID to be terminated
    :param region: Region where instance is located
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    response = client.terminate_instances(InstanceIds=[instance_id])
    if not response:
        raise Exception("Unable to terminate instance. No response received.")
    instances_terminated = response["TerminatingInstances"]
    if not instances_terminated:
        raise Exception("Failed to terminate instance.")
    if instances_terminated[0]["InstanceId"] != instance_id:
        raise Exception("Failed to terminate instance. Unknown error.")


def get_instance_type_details(instance_type, region=DEFAULT_REGION):
    """
    Get instance type details for a given instance type
    :param instance_type: Instance type to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance type
    """
    client = boto3.client("ec2", region_name=region)
    response = client.describe_instance_types(InstanceTypes=[instance_type])
    if not response or not response["InstanceTypes"]:
        raise Exception("Unable to get instance details. No response received.")
    if response["InstanceTypes"][0]["InstanceType"] != instance_type:
        raise Exception(
            f"Bad response received. Requested {instance_type} "
            f"but got {response['InstanceTypes'][0]['InstanceType']}"
        )
    return response["InstanceTypes"][0]


def get_instance_details(instance_id, region=DEFAULT_REGION):
    """
    Get instance details for instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    instance = get_instance_from_id(instance_id, region=region)
    if not instance:
        raise Exception("Could not find instance")

    return get_instance_type_details(instance["InstanceType"], region=region)


def get_ec2_fabric_connection(instance_id, instance_pem_file, region):
    """
    establish connection with EC2 instance if necessary
    :param instance_id: ec2_instance id
    :param instance_pem_file: instance key name
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    user = get_instance_user(instance_id, region=region)
    conn = Connection(
        user=user,
        host=get_public_ip(instance_id, region),
        inline_ssh_env=True,
        connect_kwargs={"key_filename": [instance_pem_file]},
    )
    return conn


def get_ec2_instance_tags(instance_id, region=DEFAULT_REGION, ec2_client=None):
    ec2_client = ec2_client or get_ec2_client(region)
    response = ec2_client.describe_tags(Filters=[{"Name": "resource-id", "Values": [instance_id]}])
    return {tag["Key"]: tag["Value"] for tag in response.get("Tags")}
