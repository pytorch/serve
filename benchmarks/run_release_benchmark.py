import boto3
import click


def get_boto_client(service_name, region_name):
    client = boto3.client(
        service_name,
        region_name=region_name,
    )
    return client


def get_boto_resource(service_name):
    client = boto3.resource(
        service_name,
        region_name='us-east-1',
    )
    return client


def start_benchmark(model_name, model_mode, batch_size, branch, ami, bucket_name, iam_role, region_name,
                    instance_type='cpu'):
    benchmark_config_path = f"preset_configs/{model_name}/{instance_type}/{model_mode}_batch_{batch_size}.json"

    s3_path = f"{model_name}/{instance_type}/{model_mode}_batch_{batch_size}"

    user_data = f'''#!/bin/bash -xe
set -eou pipefail

cleanup()
{{
aws s3api put-object --bucket {bucket_name} --key {s3_path}/exec_log.log --body /var/log/user-data.log
sudo shutdown -h now
}}

cd ~
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
pip3 install -U awscli
sudo apt-get install -y apache2-utils

trap 'cleanup;exit 1' SIGINT SIGTERM EXIT
git clone https://github.com/pytorch/serve.git
cd serve
git checkout {branch}
cd docker
./build_image.sh {"--gpu" if instance_type == "gpu" else ""} --branch_name {branch} --buildtype dev --tag pytorch/torchserve:{instance_type}
cd ../benchmarks
pip3 install -U -r requirements-ab.txt
python3 benchmark-ab.py --config {benchmark_config_path}
aws s3api put-object --bucket {bucket_name} --key {s3_path}/ab_report.csv --body /tmp/benchmark/ab_report.csv
'''

    print("Starting {} EC2 instance".format(instance_type))
    ec2_client = get_boto_client('ec2', region_name)
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
        # KeyName="torchserve", ## For Debugging change the value to your own key name.
        InstanceInitiatedShutdownBehavior='terminate',
        IamInstanceProfile={
            'Name': iam_role
        },
        UserData=user_data
    )

    print("Started {} EC2 instance".format(instance_type))
    return instances['Instances'][0]['InstanceId']


@click.command()
@click.option('--branch', '-b', default='master', help='Branch on which benchmark is to be executed. Default master')
@click.option('--instance_type', '-i', default='cpu', help='CPU/GPU instance type. Default CPU.')
@click.option('--model_name', '-mn', default='vgg11', type=click.Choice(['vgg11', 'fastrcnn', 'bert', 'vgg11_10_conc']),
              help='vgg11/fastrcnn/bert. Default vgg11')
@click.option('--model_mode', '-mm', default='eager', type=click.Choice(['eager', 'scripted']),
              help='eager/scripted. Default eager.')
@click.option('--batch_size', '-bs', default='1', type=click.Choice(['1', '2', '4', '8']), help='1/2/4/8. Default 1.')
@click.option('--ami', '-a', default='ami-02e86b825fe559330',
              help='AMI to use for EC2 instance. Default ami-02e86b825fe559330')
@click.option('--bucket_name', '-bn', required=True, help='s3 bucket name for uploading the results')
@click.option('--iam_role', '-iam', required=True, help='IAM role with access to the s3 bucket')
@click.option('--region_name', '-r', default='us-east-1', help='Region name to spin up the EC2 instance. Default us-east-1')
def auto_bench(branch, instance_type, model_name, model_mode, batch_size, ami, bucket_name, iam_role, region_name):
    ec2_instance_id = start_benchmark(model_name, model_mode, batch_size, branch, ami, bucket_name, iam_role,
                                      region_name, instance_type)
    print(ec2_instance_id)


if __name__ == '__main__':
    auto_bench()
