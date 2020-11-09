## DynamoDB Endpoint plugin
Using this plugin, you can serialize snapshots to DDB instead of files on your local file system.
Refer [plugins](README.md) for details on how to use plugins with torchserve.

You can change snapshot serializer by using a DDBEndPoint plugin as follow from default `FS` to AWS Dynamo database (DDB) -

### Assumptions
- You have aws cli installed on your machine
- Assuming you have AWS account and required privileged to create DDB tables/indexes

1. DDB serializer uses `DefaultCredentialsProvider` which supports following authorization mechanisms - 
 
    - Environment Variables - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    - Credential profiles file at the default location (~/.aws/credentials) shared by all AWS SDKs and the AWS CLI
    - Credentials delivered through the Amazon EC2 container service if AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" environment variable is set and security manager has permission to access the variable,
    - Instance profile credentials delivered through the Amazon EC2 metadata service
    
     Please configure desired auth. mechanism from above list.
     
2. Create following two tables (using aws cli)
    `aws dynamodb create-table \
    --table-name Snapshots2 \
    --attribute-definitions AttributeName=snapshotName,AttributeType=S AttributeName=createdOn,AttributeType=S AttributeName=createdOnMonth,AttributeType=S \
    --key-schema AttributeName=snapshotName,KeyType=HASH \
                AttributeName=createdOn,KeyType=RANGE \
    --provisioned-throughput ReadCapacityUnits=10,WriteCapacityUnits=5 \
    --global-secondary-indexes \
        "[
            {
                \"IndexName\": \"createdOnMonth-index\",
                \"KeySchema\": [
                    {\"AttributeName\":\"createdOnMonth\",\"KeyType\":\"HASH\"},
                    {\"AttributeName\":\"createdOn\",\"KeyType\":\"RANGE\"}
                ],
                \"Projection\": {
                    \"ProjectionType\":\"ALL\"
                },
                \"ProvisionedThroughput\": {
                    \"ReadCapacityUnits\": 10,
                    \"WriteCapacityUnits\": 5
                }
            }
        ]"`

3. Start torchserve with DDB
`torchserve --start --model-store <your-model-store-path> --plugins-path=<path-to-plugin-jars>` e.g. --plugins-path=/Users/plugins/

4. If plugin is loaded successfully then you should be able to see following -

`2020-10-26 15:06:06,705 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Trying to load snapshot serializer via plugin....
2020-10-26 15:06:06,896 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Snapshot serializer via plugin has been loaded successfully
2020-10-26 15:06:06,897 [INFO ] main DDBSnapshotSerializer - Fetching last snapshot from DDB...
2020-10-26 15:06:09,894 [ERROR] main DDBSnapshotSerializer - Failed to get last snpahost from DDB. Torchserve will start with default or given configuration.`
