#!/bin/bash
set -x
set -e


CLUSTER_NAME=TorchserveCluster
MOUNT_TARGET_GROUP_NAME="eks-efs-group"
SECURITY_GROUP_NAME="ec2-instance-group"
EC2_KEY_NAME="machine-learning"

# Fetch VPC ID / CIDR Block for the EKS Cluster
CLUSTER_NAME=TorchserveCluster
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
CIDR_BLOCK=$(aws ec2 describe-vpcs --vpc-ids $VPC_ID --query "Vpcs[].CidrBlock" --output text)

# Create a Security Group, Ingress rule
MOUNT_TARGET_GROUP_DESC="NFS access to EFS from EKS worker nodes"
MOUNT_TARGET_GROUP_ID=$(aws ec2 create-security-group --group-name $MOUNT_TARGET_GROUP_NAME --description "$MOUNT_TARGET_GROUP_DESC" --vpc-id $VPC_ID | jq --raw-output '.GroupId')
aws ec2 authorize-security-group-ingress --group-id $MOUNT_TARGET_GROUP_ID --protocol tcp --port 2049 --cidr $CIDR_BLOCK

# Create an EFS file system.
FILE_SYSTEM_ID=$(aws efs create-file-system | jq --raw-output '.FileSystemId')
aws efs describe-file-systems --file-system-id $FILE_SYSTEM_ID

sleep 10

# Create Mount target in Subnets
TAG1=tag:kubernetes.io/cluster/$CLUSTER_NAME
TAG2=tag:kubernetes.io/role/elb
subnets=($(aws ec2 describe-subnets --filters "Name=$TAG1,Values=shared" "Name=$TAG2,Values=1" | jq --raw-output '.Subnets[].SubnetId'))
for subnet in ${subnets[@]}
do
    echo "creating mount target in " $subnet
    aws efs create-mount-target --file-system-id $FILE_SYSTEM_ID --subnet-id $subnet --security-groups $MOUNT_TARGET_GROUP_ID
done

# Describes Lifecycle status of EFS File systems
aws efs describe-mount-targets --file-system-id $FILE_SYSTEM_ID | jq --raw-output '.MountTargets[].LifeCycleState'

sleep 30

SECURITY_GROUP_DESC="Allow SSH access to EC2 instance from Everywhere"
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name $SECURITY_GROUP_NAME --description "$SECURITY_GROUP_DESC" --vpc-id $VPC_ID | jq --raw-output '.GroupId')
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0

EFS_FILE_SYSTEM_DNS_NAME=$FILE_SYSTEM_ID.efs.$(aws configure get region).amazonaws.com
echo $EFS_FILE_SYSTEM_DNS_NAME
