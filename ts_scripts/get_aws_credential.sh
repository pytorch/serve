#!/bin/sh

# obtain the IAM role name
ROLE=$(curl http://169.254.169.254/latest/meta-data/iam/security-credentials)

KEYURL="http://169.254.169.254/latest/meta-data/iam/security-credentials/"$ROLE""

# Save the response into a json file
wget $KEYURL -O iam.json
 
# Use jq CLI parser to retrive the credentials: access key id; secret access key; token
 
export AWS_ACCESS_KEY_ID=$(cat iam.json | jq -r '.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(cat iam.json | jq -r '.SecretAccessKey')
