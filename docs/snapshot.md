# TorchServe model snapshot

TorchServe preserves server runtime configuration across sessions such that a TorchServe instance experiencing either a planned or unplanned service stop can restore its state upon restart.
 
A TorchServe's snapshot consists of following:

 - Server configuration, which comprises: Which models are running, which versions of those models, and how many workers are active for each model.
 - Default server configuration used while starting TorchServe the first time.
 
The snapshot is taken at following instances -

 - After successful startup, the server stores its current configuration in a timestamped snapshot file ./logs/configs/<yyyyMMddHHmmssSSS>-startup.cfg

 - If a user calls the Management API in a way that changes the server runtime config, snapshot is saved to ./logs/configs/<yyyyMMddHHmmssSSS>-snapshot.cfg

 - When the server is shut down intentionally with `torchserve --stop`, snapshot is saved to ./logs/configs/<yyyyMMddHHmmssSSS>-shutdown.cfg

User can use snapshots to restore the TorchServe's state as follows :

 - If no config file is supplied with `--ts-config-file` flag while starting TorchServe, last snapshot in ./logs/configs is used for startup.
 - If no config file is supplied with `--ts-config-file` flag and no snapshots are available, TorchServe starts with default configurations.
 - The user restarts the server specifying this config file: `torchserve --start --model-store <model store> --ts-config <known good config snapshot>`
 

If the user wishes to start without this resiliency feature, the user can start the server with :

`torchserve --start --model-store <model store> --no-config-snapshots`

This prevents to server from storing config snapshot files.

The snapshots are by default in `{LOG_LOCATION}\config` directory, where `{LOG_LOCATION}` is a system environment variable that can be used by TorchServe. If this variable is not set, the snapshot is stored in  `.\log\config` directory 

**Note** : Models passed in --models parameter while starting TorchServe are ignored if restoring from a snapshot.

# Using AWS Dynamo DB as snapshot store

Torchserve snapshots are store/serialized in local file system (FS). The snapshot serializer is responsible for saving snapshot. 
At present, torchserve supports two types of serializer, 1. `FS` 2. and `DDB` 

You can change snapshot serializer as follow  from default `FS` to AWS Dynamo database (DDB) - 
- Assuming you have AWS account and required privileged to create DDB tables

1. Create following two tables (using aws console) -
    - `latest_snapshots`
    
      `Primary key -> snapshotName`
      
    - `snapshots`
    
      `Primary key -> snapshotName`
      
      Select `Add sort key`
      
      `Sort key -> createdOn`

2. DDB serializer uses `DefaultCredentialsProvider` which supports following authorization mechanisms - 
 
    - Environment Variables - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    - Credential profiles file at the default location (~/.aws/credentials) shared by all AWS SDKs and the AWS CLI
    - Credentials delivered through the Amazon EC2 container service if AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" environment variable is set and security manager has permission to access the variable,
    - Instance profile credentials delivered through the Amazon EC2 metadata service
    
     Please configure desired auth. mechanism from above list.

3. Start torchserve with DDB
`torchserve --start --model-store <your-model-store-path> --snapshot-store DDB`

To be implemented -> Ability to supply snapshot from `snapshot` table via command line.