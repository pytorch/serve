#!/bin/bash

CLUSTER_NAME=TorchserveCluster
MOUNT_TARGET_GROUP_NAME="eks-efs-group"

# Fetch VPC ID / CIDR Block for the EKS Cluster

echo "Obtaining VPC ID for $CLUSTER_NAME"
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
echo "Obtained VPC ID - $VPC_ID"

echo "Obtaining CIDR BLOCK for $VPC_ID"
CIDR_BLOCK=$(aws ec2 describe-vpcs --vpc-ids $VPC_ID --query "Vpcs[].CidrBlock" --output text)
echo "Obtained CIDR BLOCK - $CIDR_BLOCK"


echo "Creating Security Group"
MOUNT_TARGET_GROUP_DESC="NFS access to EFS from EKS worker nodes"
MOUNT_TARGET_GROUP_ID=$(aws ec2 create-security-group --group-name $MOUNT_TARGET_GROUP_NAME --description "$MOUNT_TARGET_GROUP_DESC" --vpc-id $VPC_ID | jq --raw-output '.GroupId')
echo "Created Security Group - $MOUNT_TARGET_GROUP_ID"

echo "Configuring Security Group Ingress"
aws ec2 authorize-security-group-ingress --group-id $MOUNT_TARGET_GROUP_ID --protocol tcp --port 2049 --cidr $CIDR_BLOCK


# Create an EFS file system.
echo "Creating EFS Fils System"
FILE_SYSTEM_ID=$(aws efs create-file-system | jq --raw-output '.FileSystemId')
echo "Created EFS - $FILE_SYSTEM_ID"

aws efs describe-file-systems --file-system-id $FILE_SYSTEM_ID

echo "Waiting 30s for before procedding"
sleep 30

# Create Mount target in Subnets
echo "Obtaining Subnets"
TAG1=tag:kubernetes.io/cluster/$CLUSTER_NAME
TAG2=tag:kubernetes.io/role/elb
subnets=($(aws ec2 describe-subnets --filters "Name=$TAG1,Values=shared" "Name=$TAG2,Values=1" | jq --raw-output '.Subnets[].SubnetId'))
echo "Obtained Subnets - $subnets"


for subnet in ${subnets[@]}
do
    echo "Creating EFS Mount Target in $subnet"
    aws efs create-mount-target --file-system-id $FILE_SYSTEM_ID --subnet-id $subnet --security-groups $MOUNT_TARGET_GROUP_ID
done


EFS_FILE_SYSTEM_DNS_NAME=$FILE_SYSTEM_ID.efs.$(aws configure get region).amazonaws.com

echo "EFS File System ID - $FILE_SYSTEM_ID"
echo "EFS File System DNS Name - $EFS_FILE_SYSTEM_DNS_NAME"

echo "Succesfully created EFS & Mountpoints"
