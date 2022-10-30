# Developer Guide

The documentation covers the steps to run Torchserve along with the KServe for the mnist model in a local machine without kubernetes. This serves the purpose of developing and debugging Kserve wrapper, service envelope for Torchserve.

## Prerequisites

Below are the prerequisites should be met.

- Torchserve >= 0.6.0
- Torch-model-archiver >= 0.6.0 
- Kserve >= 0.8.0

## Steps to run Torchserve with Kserve

### Generating marfile and config.properties file

Navigate to the cloned serve repo and run

```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```

The above command generates mnist_kf.mar

Copy the below contents to config.properties and change the model_store path

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=<path-to-the-mar-file>
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist_kf":{"1.0":{"defaultVersion":true,"marName":"mnist_kf.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```

### Running Torchserve

Enable KServe service envelope

```bash
# For v1 protocol 
export TS_SERVICE_ENVELOPE=kserve
# For v2 protocol
export TS_SERVICE_ENVELOPE=kservev2
```

Start Torchserve

```bash
torchserve --start --ts-config config.properties
```

Output

```bash
2022-08-22T12:47:40,241 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Initializing plugins manager...                                                                
2022-08-22T12:47:40,344 [INFO ] main org.pytorch.serve.ModelServer -                                                                                                                   
Torchserve version: 0.5.3                                                                                                                                                              
TS Home: /home/ubuntu/.virtualenvs/base/lib/python3.8/site-packages                                                                                                                    
Current directory: /home/ubuntu/Repositories                                                                                                                                        
Temp directory: /tmp                                                                                                                                                                   
Number of GPUs: 0                                                                                                                                                                      
Number of CPUs: 8                                                                                                                                                                      
Max heap size: 3922 M                                                                                                                                                                  
Python executable: /home/ubuntu/.virtualenvs/base/bin/python                                                                                                                           
Config file: ./serve/kubernetes/kserve/config.properties                                                                                                                        
Inference address: http://0.0.0.0:8085                                                                                                                                                 
Management address: http://0.0.0.0:8085                                                                                                                                                
Metrics address: http://0.0.0.0:8082                                                                                                                                                   
Model Store: /home/ubuntu/Repositories/fb/serve                                                                                                                                        
Initial Models: N/A                                                                                                                                                                    
Log dir: /home/ubuntu/Repositories/fb/logs                                                                                                                                             
Metrics dir: /home/ubuntu/Repositories/fb/logs                                                                                                                                         
Netty threads: 4                                                                                                                                                                       
Netty client threads: 0                                                                                                                                                                
Default workers per model: 8                                                                                                                                                           
Blacklist Regex: N/A                                                                                                                                                                   
Maximum Response Size: 6553500                                                                                                                                                         
Maximum Request Size: 6553500                                                                                                                                                          
Limit Maximum Image Pixels: true                                                                                                                                                       
Prefer direct buffer: false                                                                                                                                                            
Allowed Urls: [file://.*|http(s)?://.*]                                                                                                                                                
Custom python dependency for model allowed: true                                                                                                                                       
Metrics report format: prometheus                                                                                                                                                      
Enable metrics API: true                                                                                                                                                               
Workflow Store: /home/ubuntu/Repositories/fb/serve
Model config: N/A
2022-08-22T12:47:40,363 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager -  Loading snapshot serializer plugin...
2022-08-22T12:47:40,391 [INFO ] main org.pytorch.serve.snapshot.SnapshotManager - Started restoring models from snapshot {"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":5,"batchSize":5,"maxBatchDelay":200,"responseTimeout":60}}}}
2022-08-22T12:47:40,399 [INFO ] main org.pytorch.serve.snapshot.SnapshotManager - Validating snapshot startup.cfg
2022-08-22T12:47:40,400 [INFO ] main org.pytorch.serve.snapshot.SnapshotManager - Snapshot startup.cfg validated successfully
2022-08-22T12:47:40,511 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Adding new version 1.0 for model mnist
2022-08-22T12:47:40,511 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model mnist
2022-08-22T12:47:40,512 [DEBUG] main org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model mnist
2022-08-22T12:47:40,512 [INFO ] main org.pytorch.serve.wlm.ModelManager - Model mnist loaded.
2022-08-22T12:47:40,512 [DEBUG] main org.pytorch.serve.wlm.ModelManager - updateModel: mnist, count: 1
2022-08-22T12:47:40,521 [DEBUG] W-9000-mnist_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - Worker cmdline: [/home/ubuntu/.virtualenvs/base/bin/python, /home/ubuntu/.virtualenvs/base/lib/python3.8/site-packages/ts/model_service_worker.py, --sock-type, unix, --sock-name, /tmp/.ts.sock.9000]
2022-08-22T12:47:40,523 [INFO ] main org.pytorch.serve.ModelServer - Initialize Inference server with: EpollServerSocketChannel.
```

### Running KServe


Start KServe Serve

Navigate to serve/kubernetes/kserve/kserve_wrapper

```bash
python3 __main__.py
```

Output

```bash
[I 220822 12:44:05 __main__:75] Wrapper : Model names ['mnist'], inference address http//0.0.0.0:8085, management address http://0.0.0.0:8085, model store /mnt/models/model-store
[I 220822 12:44:05 TorchserveModel:54] kfmodel Predict URL set to 0.0.0.0:8085
[I 220822 12:44:05 TorchserveModel:56] kfmodel Explain URL set to 0.0.0.0:8085
[I 220822 12:44:05 TorchserveModel:54] kfmodel Predict URL set to 0.0.0.0:8085
[I 220822 12:44:05 TorchserveModel:56] kfmodel Explain URL set to 0.0.0.0:8085
[I 220822 12:44:05 TSModelRepository:30] TSModelRepo is initialized
[I 220822 12:44:05 model_server:150] Registering model: mnist
[I 220822 12:44:05 model_server:123] Listening on port 8080
[I 220822 12:44:05 model_server:125] Will fork 1 workers
[I 220822 12:44:05 model_server:128] Setting max asyncio worker threads as 12
```

### Make inference request

```bash
# v1 protocol
cd serve/kubernetes/kserve/kf_request_json/v1
# Infer request
curl http://localhost:8080/v1/models/mnist:predict -d @./mnist.json
```

```json
{"predictions": [2]}
```

```bash
# v2 protocol
cd serve/kubernetes/kserve/kf_request_json/v2/mnist
# Infer request
curl http://localhost:8080/v2/models/mnist/infer -d @./mnist_v2_bytes.json
```

```json
{"id": "7a02adc8-e1f2-4218-a2ad-f4a29dfa9b16", "model_name": "mnist", "model_version": "1.0", "outputs": [{"name": "predict", "shape": [], "datatype": "INT64", "data": [0]}]}
```

### Make explain request

```bash
# v1 protocol
cd serve/kubernetes/kserve/kf_request_json/v1
# Explain request
curl http://localhost:8080/v1/models/mnist:explain -d @./mnist.json
```

```json
{"explanations": [[[[0.004570948731989492, 0.006216969640322402, 0.008197565423679522, 0.009563574612830427, 0.008999274832810742, 0.009673474804303854, 0.007599905146155397, 0.00636138087221357, 0.005768828729217899, 0.004394465747976554, 0.004948218056579564, 0.005273460629510146, 0.005523799690682735, 0.007789356618988726, 0
.
.
.
0015855262453375175, 0.0036409566641309194, -0.0005390934328924084, 0.0064513327328557446, 0.0027735805355367277, 0.006060840367244276, 0.000359165926315527, 0.0018643897471817563, -0.0008303191079628302, -0.0024594973537835877, -0.0017738576926115562, -0.0007076670305583287, -0.0], [-0.0, -0.0, 0.0005382485914412182, -0.0006324885664017992, -0.003595975043089762, -0.0018980114805717792, -0.0005749303948698967, 0.0012654955920759015, 0.0036969897692216216, 0.000977110922868114, -0.0003167563284092459, -0.0005752726948934128, 0.000789864852287303, 0.002705500072887267, -0.00040143920665009533, 0.0011896338595401026, 0.0002241121980320825, -0.00025305534684833724, -6.698087605655334e-05, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]]]]}
```

```bash
# v2 protocol
cd serve/kubernetes/kserve/kf_request_json/v2/mnist
# Explain request
curl http://localhost:8080/v1/models/mnist/explain -d @./mnist_v2_bytes.json
```

```json
{"id": "67ebd591-9528-4172-8aef-e500bc45112c", "model_name": "mnist", "model_version": "1.0", "outputs": [{"name": "explain", "shape": [1, 28, 28], "datatype": "FP64", "data": [-0.0003991945059908792, -0.0001900219080570986, -0.0008597191651461465, -0.000329367249820099, -0.0009114924783138575, -0.00017816117412418168, -0.0005801028955959462, -5.752924725363697e-05, -0.00013036395115086704, -3.622335859118503e-05, 1.2628162842923433e-05, -3.151632502594692e-05, -7.058361567244847e-05, -6.872769525033832e-05, -7.877430620392666e-05, -0.00013302445762245664, -3.925598302192034e-05, -0.00022929557168679322, 1.7131089672114954e-05, -0.00038463532100758497, -0.0005095514761288278, -0.00039880108606847803, -0.000845961881996372, -0.0005034791333543297, -0.0006029826446001493, -0.
.
.
.
004584260323458741, -0.00020735189345109017, -0.00043539764473153527, -0.00013556153451301715, -0.00017089609432996696, -9.525744923599118e-05, 0.0005243356345696903, 7.008281799400381e-05, 7.51456093974107e-05, -0.0005560118982914238, -0.00023047035476863136, -0.0003051983831035168, -0.00022037598447062156, -0.0001406931007901416, -0.00010127583504453587, 7.5947113286326e-05, 0.0002917843269490143, 9.378577280957779e-05, -0.0, -0.0, -0.0, -0.0, -0.0, -2.1141910767920675e-05, 3.0694499085548986e-05, -0.00027611975917940344, 0.00013554868268694786, 8.26728299954e-05, -0.00018739746111324474, -0.00023583001888983061, -3.794037105817584e-05, 0.00013116482494314208, 0.00013064037871521623, 1.711969851912632e-05, 0.000470505263982109, 0.0001437191226144529, 0.00011352509966634157, -0.0002483073596307892, -0.00010922927830782495, -8.497594950123566e-05, -6.623092873323381e-05, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0]}]}
````