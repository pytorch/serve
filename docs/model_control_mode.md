# Model Control Mode

TorchServe now supports model control mode with two settings "none"(default) and "explicit"

## Two ways to set Model Control
1. Add `--model-mode-explicit` to command line when running TorchServe to switch from none to explicit mode. Command line cannot be used to set mode to none, can only be used to set to explicit
2. Add `model_control_mode=explicit` or `model_control_mode=none` to config.properties file

Priority between cmd and config file follows the following [TorchServer standard](https://github.com/pytorch/serve/blob/c74a29e8144bc12b84196775076b0e8cf3c5a6fc/docs/configuration.md#advanced-configuration)
* Example 1:
  * Config file: `model_control_mode=none`

    cmd line: `torchserve --start --ncs --model-store model_store --model-mode-explicit`

    Result: Explicit mode enabled
* Example 2:
  * Config file: `model_control_mode=explicit`

    cmd line: `torchserve --start --ncs --model-store model_store`

    Result: Mode is explicit (no way to enable "none" through cmd)

## Model Control Mode Default
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

## Model Control Mode EXPLICIT
Setting model control to `explicit` allows users to load and unload models using the model load APIs.

### Example using cmd line to set mode to explicit
```
ubuntu@ip-172-31-11-32:~/serve$ torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --ts-config config.properties --model-mode-explicit

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
