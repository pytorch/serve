# TorchServe model snapshot

TorchServe preserves server runtime configuration across sessions such that a TorchServe instance experiencing either a planned or unplanned service stop can restore its state upon restart.
 
__Note:__ Current snapshot does not support workflow.

A TorchServe's snapshot consists of following:

 - Server configuration, which comprises: Which models are running, which versions of those models, and how many workers are active for each model.
 - Default server configuration used while starting TorchServe the first time.
 
The snapshot is taken at following instances -

 - After successful startup, the server stores its current configuration in a timestamped snapshot file ./logs/config/<yyyyMMddHHmmssSSS>-startup.cfg

 - If a user calls the Management API in a way that changes the server runtime config, snapshot is saved to ./logs/config/<yyyyMMddHHmmssSSS>-snapshot.cfg

 - When the server is shut down intentionally with `torchserve --stop`, snapshot is saved to ./logs/configs/<yyyyMMddHHmmssSSS>-shutdown.cfg

User can use snapshots to restore the TorchServe's state as follows :

 - If no config file is supplied with `--ts-config-file` flag while starting TorchServe, last snapshot in ./logs/configs is used for startup.
 - If no config file is supplied with `--ts-config-file` flag and no snapshots are available, TorchServe starts with default configurations.
 - The user restarts the server specifying this config file: `torchserve --start --model-store <model store> --ts-config <known good config snapshot>`
 

If the user wishes to start without this resiliency feature, the user can start the server with :

`torchserve --start --model-store <model store> --no-config-snapshots`

This prevents to server from storing config snapshot files.

The snapshots are by default in `{LOG_LOCATION}\config` directory, where `{LOG_LOCATION}` is a system environment variable that can be used by TorchServe. If this variable is not set, the snapshot is stored in  `.\log\config` directory 

**Note** : 
1. Models passed in --models parameter while starting TorchServe are ignored if restoring from a snapshot.
2. For windows, if shutdown snapshot file is not generated then you can use last snapshot file.
