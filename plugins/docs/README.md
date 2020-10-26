## Torchserve plugins

You can create following types of plugins for torchserve to customize the related behaviour.

1. Endpoint apis - If you want to add additional apis for your use case then you can do so by adding apis
of following type using plugin. e.g. [endpoints plugin](../plugins/endpoints/)
    a. management api
    b. inference api
    c. metric api
    
2. Snapshot serializer - It is possible to override the default file based serializer of torchserve. For example,
here is [AWS DynamoDB snapshot serializer](../plugins/DDBEndPoint). This enables torchserve to serialize snapshots to DynamoDB. 

### How to use plugins with torchserve. 
There are following two ways to inluce plugin jars to torchserve.

1. Using config. property - `plugins_path` 
e.g.
Add following line to your torchserve config. properties file.
`plugins_path=<path-containing-plugin-jars">

2. Using command line option  `--plugins-path`
e.g. 
`torchserve --start --model-store <your-model-store-path> --plugins-path=<path-to-plugin-jars>` 

e.g. --plugins-path=/Users/plugins/