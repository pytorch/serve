import boto3
import click
import os
import paramiko
import time


def get_boto_client(service_name):
    client = boto3.client(
        service_name,
        region_name='us-east-1',
    )
    return client


def get_boto_resource(service_name):
    client = boto3.resource(
        service_name,
        region_name='us-east-1',
    )
    return client


def start_ec2_instance(key_name, ami, security_group, subnet_id, instance_type='cpu'):
    print("Starting {} EC2 instance".format(instance_type))
    ec2_client = get_boto_client('ec2')
    instances = ec2_client.run_instances(
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sdh',
                'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': 200,
                    'VolumeType': 'standard',
                },
            },
        ],
        ImageId=ami,
        InstanceType='c4.4xlarge' if instance_type == "cpu" else "p3.8xlarge",
        KeyName=key_name,
        MaxCount=1,
        MinCount=1,
        SecurityGroupIds=[
            security_group,
        ],
        SubnetId=subnet_id,
        InstanceInitiatedShutdownBehavior='terminate',
    )

    print("Started {} EC2 instance".format(instance_type))
    return instances['Instances'][0]['InstanceId']


def get_instance_meta(ec2_instance_id):
    ec2_client = get_boto_client('ec2')
    instance_status = ec2_client.describe_instance_status(
        InstanceIds=[
            ec2_instance_id
        ],
    )

    print('EC2 instance initializing... Checking status')
    while len(instance_status['InstanceStatuses']) == 0:
        time.sleep(2)
        instance_status = ec2_client.describe_instance_status(
            InstanceIds=[
                ec2_instance_id
            ],
        )

    while (not (instance_status['InstanceStatuses'][0]['InstanceStatus']['Status'] == 'ok'
                and instance_status['InstanceStatuses'][0]['SystemStatus']['Status'] == 'ok')):
        print("Instance not ready retrying in 15 seconds")
        time.sleep(10)
        instance_status = ec2_client.describe_instance_status(
            InstanceIds=[
                ec2_instance_id
            ],
        )

    instance_meta = ec2_client.describe_instances(
        InstanceIds=[
            ec2_instance_id
        ],
    )
    print("Public IP address is : {}".format(instance_meta['Reservations'][0]['Instances'][0]['PublicIpAddress']))
    return instance_meta['Reservations'][0]['Instances'][0]['PublicIpAddress']


def run_benchmark(key_file, public_ip_address, branch, model_name, model_mode, batch_size, instance_type):
    # while not stdout.channel.exit_status_ready():
    print("Creating remote ssh connection to EC2 instance...")
    key = paramiko.RSAKey.from_private_key_file(key_file)

    client = paramiko.SSHClient()

    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(hostname=public_ip_address, username="ubuntu", pkey=key)

    print("Connected to remote EC2 instance...")
    print("Cloning serve repo...")
    command = 'git clone https://github.com/pytorch/serve.git && cd serve && git checkout {}'.format(branch)

    print("Executing command  : {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)

    exit_status = stdout.channel.recv_exit_status()

    if exit_status != 0:
        print(stdout.read())
        print(stderr.read())

    print("TorchServe repo cloned.")
    print("Creating docker image...")
    command = 'cd serve/docker && ./build_image.sh' \
              ' {docker_type} --branch_name {branch_name} --tag pytorch/torchserve:{tag_name}' \
        .format(docker_type="--gpu" if instance_type == "gpu" else "",
                branch_name=branch,
                tag_name=instance_type
                )
    print("Executing command  : {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()

    if exit_status != 0:
        print(stdout.read())
        print(stderr.read())

    print("Docker image creation completed.")
    print("Installing benchmark dependencies...")
    command = 'cd serve/benchmarks' \
              ' && pip install -U -r requirements-ab.txt' \
              ' && sudo apt-get install -y apache2-utils'
    print("Executing command  : {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(stdout.read())
        print(stderr.read())

    benchmark_config_path = "preset_configs/{model_name}/{instance_type}/{model_mode}_batch_{batch_size}.json".format(
        model_name=model_name,
        instance_type=instance_type,
        model_mode=model_mode,
        batch_size=batch_size
    )

    print("Benchmark dependency installation completed.")
    print("Running benchmark...")
    command = 'cd serve/benchmarks && python benchmark-ab.py --config {}'.format(benchmark_config_path)
    print("Executing command  : {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()

    if exit_status != 0:
        print(stdout.read())
        print(stderr.read())

    print("Benchmark execution completed... Collecting results...")
    command = 'cat /tmp/benchmark/ab_report.csv'
    print("Executing command  : {}".format(command))
    stdin, stdout, stderr = client.exec_command(command)

    report_path = '/tmp/benchmark'
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    report_file = '{}/{}_{}_{}.csv'.format(report_path, model_name, model_mode, batch_size)

    with open(report_file, 'w') as rf:
        for line in stdout.readlines():
            rf.write(line)

    client.close()


def terminate_ec2_instance(ec2_instance_id):
    print("terminating ec2 instance")
    ec2_client = get_boto_client('ec2')
    response = ec2_client.terminate_instances(
        InstanceIds=[
            ec2_instance_id
        ],
    )

    print("EC2 instance terminated")


@click.command()
@click.option('--branch', '-b', default='master', help='Branch on which benchmark is to be executed. Default master')
@click.option('--instance_type', '-i', default='cpu', help='CPU/GPU instance type. Default CPU.')
@click.option('--model_name', '-mn', default='vgg11', type=click.Choice(['vgg11', 'fastrcnn', 'bert']),
              help='vgg11/fastrcnn/bert. Default vgg11')
@click.option('--model_mode', '-bs', default='eager',type=click.Choice(['eager', 'scripted']),
              help='eager/scripted. Default eager.')
@click.option('--batch_size', '-bs', default=1, type=click.Choice([1, 2, 4, 8]), help='1/2/4/8. Default 1.')
@click.option('--ami', '-a', default='ami-079d181e97ab77906',
              help='AMI to use for EC2 instance. Default ami-079d181e97ab77906')
@click.option('--ec2_key_file', '-k', required=True, help='Path to pem file to be used for instantiating EC2')
@click.option('--subnet_id', '-s', required=True, help='Subnet ID to use for EC2 instance')
@click.option('--security_group_id', '-sg', required=True, help='Security Group ID to use for EC2 instance')
def auto_bench(branch, instance_type, model_name, model_mode, batch_size, ami, ec2_key_file, subnet_id, security_group_id):
    key_name = ec2_key_file.split("/")[-1].split(".")[0]
    ec2_instance_id = start_ec2_instance(key_name, ami, security_group_id, subnet_id, instance_type)
    public_ip_address = get_instance_meta(ec2_instance_id)
    run_benchmark(ec2_key_file, public_ip_address, branch, model_name, model_mode, batch_size, instance_type)
    terminate_ec2_instance(ec2_instance_id)


if __name__ == '__main__':
    auto_bench()
