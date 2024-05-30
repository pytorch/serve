# Model Control Mode

TorchServe now supports model control mode with two settings "none"(default) and "explicit"

## How to set Model Control to explicit
1. Add `--model-mode-explicit` to command line when running TorchServe.
2. Add `model_control_mode=explicit` to config.properties file

## Model Control Mode Default
At startup TorchServe loads only those models specified explicitly with the `--models` command-line option. After startup users will be unable to register or delete models in this mode.

### Example default
```
ubuntu@ip-172-31-11-32:~/serve$ torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --ts-config config.properties
ubuntu@ip-172-31-11-32:~/serve$ WARNING: sun.reflect.Reflection.getCallerClass is not supported. This will impact performance.
2024-05-30T21:45:48,383 [WARN ] main org.pytorch.serve.util.ConfigManager - Your torchserve instance can access any URL to load models. When deploying to production, make sure to limit the set of allowed_urls in config.properties
2024-05-30T21:45:48,386 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Initializing plugins manager...
2024-05-30T21:45:48,445 [INFO ] main org.pytorch.serve.metrics.configuration.MetricConfiguration - Successfully loaded metrics configuration from /opt/conda/lib/python3.10/site-packages/ts/configs/metrics.yaml
2024-05-30T21:45:48,565 [INFO ] main org.pytorch.serve.ModelServer -
Torchserve version: 0.11.0
TS Home: /opt/conda/lib/python3.10/site-packages
Current directory: /home/ubuntu/serve
Temp directory: /tmp
Metrics config path: /opt/conda/lib/python3.10/site-packages/ts/configs/metrics.yaml
Number of GPUs: 0
Number of CPUs: 4
Max heap size: 3924 M
Python executable: /opt/conda/bin/python3
Config file: config.properties
Inference address: http://127.0.0.1:8080
Management address: http://127.0.0.1:8081
Metrics address: http://127.0.0.1:8082
Model Store: /home/ubuntu/serve/model_store
Initial Models: resnet-18=resnet-18.mar
Log dir: /home/ubuntu/serve/logs
Metrics dir: /home/ubuntu/serve/logs
Netty threads: 0
Netty client threads: 0
Default workers per model: 4
Blacklist Regex: N/A
Maximum Response Size: 6553500
Maximum Request Size: 6553500
Limit Maximum Image Pixels: true
Prefer direct buffer: false
Allowed Urls: [file://.*|http(s)?://.*]
Custom python dependency for model allowed: false
Enable metrics API: true
Metrics mode: LOG
Disable system metrics: false
Workflow Store: /home/ubuntu/serve/model_store
CPP log config: N/A
Model config: N/A
System metrics command: default
Model control mode: default
Model server started.
...
ubuntu@ip-172-31-11-32:~/serve$ curl -X POST  "http://localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"
2024-05-30T21:46:03,625 [INFO ] epollEventLoopGroup-3-2 ACCESS_LOG - /127.0.0.1:53514 "POST /models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar HTTP/1.1" 405 0
2024-05-30T21:46:03,626 [INFO ] epollEventLoopGroup-3-2 TS_METRICS - Requests4XX.Count:1.0|#Level:Host|#hostname:ip-172-31-11-32,timestamp:1717105563
{
  "code": 405,
  "type": "MethodNotAllowedException",
  "message": "Requested method is not allowed, please refer to API document."
}
ubuntu@ip-172-31-11-32:~/serve$ curl http://localhost:8081/models
2024-05-30T21:46:06,450 [INFO ] epollEventLoopGroup-3-3 ACCESS_LOG - /127.0.0.1:53516 "GET /models HTTP/1.1" 200 3
2024-05-30T21:46:06,450 [INFO ] epollEventLoopGroup-3-3 TS_METRICS - Requests2XX.Count:1.0|#Level:Host|#hostname:ip-172-31-11-32,timestamp:1717105566
{
  "models": [
    {
      "modelName": "resnet-18",
      "modelUrl": "resnet-18.mar"
    }
  ]
}
ubuntu@ip-172-31-11-32:~/serve$ torchserve --stop
TorchServe has stopped.
```

## Model Control Mode EXPLICIT
Setting model control to `explicit` allows users to load and unload models using the model load APIs.

### Example using cmd line to set mode to explicit
```
ubuntu@ip-172-31-11-32:~/serve$ torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --ts-config config.properties --model-mode-explicit
ubuntu@ip-172-31-11-32:~/serve$ WARNING: sun.reflect.Reflection.getCallerClass is not supported. This will impact performance.
2024-05-30T21:41:14,431 [WARN ] main org.pytorch.serve.util.ConfigManager - Your torchserve instance can access any URL to load models. When deploying to production, make sure to limit the set of allowed_urls in config.properties
2024-05-30T21:41:14,433 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Initializing plugins manager...
2024-05-30T21:41:14,490 [INFO ] main org.pytorch.serve.metrics.configuration.MetricConfiguration - Successfully loaded metrics configuration from /opt/conda/lib/python3.10/site-packages/ts/configs/metrics.yaml
2024-05-30T21:41:14,605 [INFO ] main org.pytorch.serve.ModelServer -
Torchserve version: 0.11.0
TS Home: /opt/conda/lib/python3.10/site-packages
Current directory: /home/ubuntu/serve
Temp directory: /tmp
Metrics config path: /opt/conda/lib/python3.10/site-packages/ts/configs/metrics.yaml
Number of GPUs: 0
Number of CPUs: 4
Max heap size: 3924 M
Python executable: /opt/conda/bin/python3
Config file: config.properties
Inference address: http://127.0.0.1:8080
Management address: http://127.0.0.1:8081
Metrics address: http://127.0.0.1:8082
Model Store: /home/ubuntu/serve/model_store
Initial Models: resnet-18=resnet-18.mar
Log dir: /home/ubuntu/serve/logs
Metrics dir: /home/ubuntu/serve/logs
Netty threads: 0
Netty client threads: 0
Default workers per model: 4
Blacklist Regex: N/A
Maximum Response Size: 6553500
Maximum Request Size: 6553500
Limit Maximum Image Pixels: true
Prefer direct buffer: false
Allowed Urls: [file://.*|http(s)?://.*]
Custom python dependency for model allowed: false
Enable metrics API: true
Metrics mode: LOG
Disable system metrics: false
Workflow Store: /home/ubuntu/serve/model_store
CPP log config: N/A
Model config: N/A
System metrics command: default
Model control mode: explicit
2024-05-30T21:41:14,612 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager -  Loading snapshot serializer plugin...
2024-05-30T21:41:14,635 [INFO ] main org.pytorch.serve.ModelServer - Loading initial models: resnet-18.mar
2024-05-30T21:41:15,465 [INFO ] main org.pytorch.serve.wlm.ModelManager - Model resnet-18 loaded.
Model server started.
...
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
