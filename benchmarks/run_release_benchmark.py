import boto3
import click
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


def start_benchmark(model_name, model_mode, batch_size, branch, ami, security_group, subnet_id, bucket_name,
                    iam_role, instance_type='cpu'):
    benchmark_config_path = f"preset_configs/{model_name}/{instance_type}/{model_mode}_batch_{batch_size}.json"

    report_key = f"{model_name}/{instance_type}/{model_mode}_batch_{batch_size}/ab_report.csv"

    user_data = f'''#!/bin/bash -xe
cd ~
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
git clone https://github.com/pytorch/serve.git
cd serve
git checkout {branch}
cd docker
./build_image.sh {"--gpu" if instance_type == "gpu" else ""} --branch_name {branch} --buildtype dev --tag pytorch/torchserve:{instance_type}
cd ../benchmarks
pip3 install -U -r requirements-ab.txt
pip3 install -U awscli
sudo apt-get install -y apache2-utils
python3 benchmark-ab.py --config {benchmark_config_path}
aws s3api put-object --bucket {bucket_name} --key {report_key} --body /tmp/benchmark/ab_report.csv
sudo shutdown -h now
'''

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
        MaxCount=1,
        MinCount=1,
        SecurityGroupIds=[
            security_group,
        ],
        SubnetId=subnet_id,
        InstanceInitiatedShutdownBehavior='terminate',
        IamInstanceProfile={
            'Name': iam_role
        },
        UserData=user_data
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


@click.command()
@click.option('--branch', '-b', default='master', help='Branch on which benchmark is to be executed. Default master')
@click.option('--instance_type', '-i', default='cpu', help='CPU/GPU instance type. Default CPU.')
@click.option('--model_name', '-mn', default='vgg11', type=click.Choice(['vgg11', 'fastrcnn', 'bert']),
              help='vgg11/fastrcnn/bert. Default vgg11')
@click.option('--model_mode', '-mm', default='eager', type=click.Choice(['eager', 'scripted']),
              help='eager/scripted. Default eager.')
@click.option('--batch_size', '-bs', default='1', type=click.Choice(['1', '2', '4', '8']), help='1/2/4/8. Default 1.')
@click.option('--ami', '-a', default='ami-02e86b825fe559330',
              help='AMI to use for EC2 instance. Default ami-02e86b825fe559330')
@click.option('--subnet_id', '-s', required=True, help='Subnet ID to use for EC2 instance')
@click.option('--security_group_id', '-sg', required=True, help='Security Group ID to use for EC2 instance')
@click.option('--bucket_name', '-bn', required=True, help='s3 bucket name for uploading the results')
@click.option('--iam_role', '-iam', required=True, help='IAM role with access to the s3 bucket')
def auto_bench(branch, instance_type, model_name, model_mode, batch_size, ami, subnet_id,
               security_group_id, bucket_name, iam_role):
    ec2_instance_id = start_benchmark(model_name, model_mode, batch_size, branch, ami, security_group_id,
                                      subnet_id, bucket_name, iam_role, instance_type)
    #public_ip_address = get_instance_meta(ec2_instance_id)


if __name__ == '__main__':
    auto_bench()
