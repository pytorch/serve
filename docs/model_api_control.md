# Model API Control

TorchServe now supports model API control with two settings enabled and disabled(default).

TorchServe introduces a new feature called model-api-control which allow users to prevent registering and deleting models once the servers are running. This is a security feature which addresses the concern of unwanted registering and deleting of models once the TorchServe servers have started. This is applicable in the scenario where a user may upload malicious code to the model server in the form of a model or where a user may delete a model that is being used. The default behavior prevents users from registering or deleting models once TorchServe is running, and then you can enable the model APIs to allow users to register and delete models whenever using the TorchServe model load APIs.

## Two ways to set Model Control
1. Add `--model-api-enabled` to command line when running TorchServe to switch from disabled to enabled. Command line cannot be used to disabled, can only be used to enabled
2. Add `model_api_enabled=false` or `model_api_enabled=true` to config.properties file
    * `model_api_enabled=false` is default and prevents users from registering or deleting models once TorchServe is running
    * `model_api_enabled=true` is not default and allows users to register and delete models using the TorchServe model load APIs

Priority between cmd and config file follows the following [TorchServer standard](https://github.com/pytorch/serve/blob/c74a29e8144bc12b84196775076b0e8cf3c5a6fc/docs/configuration.md#advanced-configuration)
* Example 1:
  * Config file: `model_api_enabled=false`

    cmd line: `torchserve --start --ncs --model-store model_store --model-api-enabled`

    Result: Model api mode enabled
* Example 2:
  * Config file: `model_api_enabled=true`

    cmd line: `torchserve --start --ncs --model-store model_store`

    Result: Mode is enabled (no way to disable api mode through cmd)

## Model API Control Default
At startup TorchServe loads only those models specified explicitly with the `--models` command-line option. After startup users will be unable to register or delete models in this mode.

### Example default
```
ubuntu@ip-172-31-11-32:~/serve$ torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --ts-config config.properties
...
ubuntu@ip-172-31-11-32:~/serve$ curl -X POST  "http://localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"
2024-05-30T21:46:03,625 [INFO ] epollEventLoopGroup-3-2 ACCESS_LOG - /127.0.0.1:53514 "POST /models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar HTTP/1.1" 405 0
2024-05-30T21:46:03,626 [INFO ] epollEventLoopGroup-3-2 TS_METRICS - Requests4XX.Count:1.0|#Level:Host|#hostname:ip-172-31-11-32,timestamp:1717105563
{
  "code": 405,
  "type": "MethodNotAllowedException",
  "message": "Requested method is not allowed, please refer to API document."
}
```

## Model Control API Enabled
Setting model API to `enabled` allows users to load and unload models using the model load APIs.

### Example using cmd line to set mode to enabled
```
ubuntu@ip-172-31-11-32:~/serve$ torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --ts-config config.properties --model-api-enabled

ubuntu@ip-172-31-11-32:~/serve$ curl -X POST  "http://localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"
{
  "status": "Model \"squeezenet1_1\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
}
ubuntu@ip-172-31-11-32:~/serve$ curl http://localhost:8081/models
2024-05-30T21:41:47,098 [INFO ] epollEventLoopGroup-3-2 ACCESS_LOG - /127.0.0.1:36270 "GET /models HTTP/1.1" 200 2
2024-05-30T21:41:47,099 [INFO ] epollEventLoopGroup-3-2 TS_METRICS - Requests2XX.Count:1.0|#Level:Host|#hostname:ip-172-31-11-32,timestamp:1717105307
{
  "models": [
    {
      "modelName": "resnet-18",
      "modelUrl": "resnet-18.mar"
    },
    {
      "modelName": "squeezenet1_1",
      "modelUrl": "https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"
    }
  ]
}
ubuntu@ip-172-31-11-32:~/serve$ torchserve --stop
TorchServe has stopped.
```
