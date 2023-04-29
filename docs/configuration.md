# Advanced configuration

The default settings form TorchServe should be sufficient for most use cases. However, if you want to customize TorchServe, the configuration options described in this topic are available.

There are three ways to configure TorchServe. In order of priority, they are:

1. Environment variables
2. Command line arguments
3. Configuration file

For example, the value of an environment variable overrides both command line arguments and
a property in the configuration file. The value of a command line argument overrides
a value in the configuration file.

## Environment variables

You can change TorchServe behavior by setting the following environment variables:

* JAVA_HOME
* PYTHONPATH
* TS_CONFIG_FILE
* LOG_LOCATION
* METRICS_LOCATION

**Note:** Environment variables have higher priority than command line or config.properties.
The value of an environment variable overrides other property values.

## Command line parameters

Customize TorchServe behavior by using the following command line arguments when you call `torchserve`:

* **--ts-config** TorchServe loads the specified configuration file if `TS_CONFIG_FILE` environment variable is not set
* **--model-store** Overrides the `model_store` property in config.properties file
* **--models** Overrides the `load_models` property in config.properties
* **--log-config** Overrides the default log4j2.xml
* **--foreground** Runs TorchServe in the foreground. If this option is
                        disabled, TorchServe runs in the background

For more detailed information about `torchserve` command line options, see [Serve Models with TorchServe](server.md).

## config.properties file

TorchServe uses a `config.properties` file to store configurations. TorchServe uses following, in order of priority, to locate this `config.properties` file:

1. If the `TS_CONFIG_FILE` environment variable is set, TorchServe loads the configuration from the path specified by the environment variable.
2. If `--ts-config` parameter is passed to `torchserve`, TorchServe loads the configuration from the path specified by the parameter.
3. If there is a `config.properties` in the folder where you call `torchserve`, TorchServe loads the `config.properties` file from the current working directory.
4. If none of the above is specified, TorchServe loads a built-in configuration with default values.

### Customize JVM options

To control TorchServe frontend memory footprint, configure the **vmargs** property in the `config.properties` file

* default: N/A, use JVM default options

Adjust JVM options to fit your memory requirement.

### Load models at startup

You can configure TorchServe to load models during startup by setting the `model_store` and `load_models` properties.
The following values are valid:

* `load_models`
  * `standalone`: default: N/A, No models are loaded at start up.
  * `all`: Load all models present in `model_store`.
  * `model1.mar, model2.mar`: Load models in the specified MAR files from `model_store`.
  * `model1=model1.mar, model2=model2.mar`: Load models with the specified names and MAR files from `model_store`.

* `model_store`
  * `standalone`: default: N/A, Loading models from the local disk is disabled.
  * `pathname`: The model store location is specified by the value of `pathname`.

**Note:** `model_store` and `load_models` properties are overridden by command line parameters, if specified.

### Configure TorchServe listening address and port

TorchServe doesn't support authentication natively. To avoid unauthorized access, TorchServe only allows localhost access by default.
The inference API is listening on port 8080. The management API is listening on port 8081. Both expect HTTP requests. These are the default ports.
See [Enable SSL](#enable-ssl) to configure HTTPS.

* `inference_address`: Inference API binding address. Default: `http://127.0.0.1:8080`
* `management_address`: Management API binding address. Default: `http://127.0.0.1:8081`
* `metrics_address`: Metrics API binding address. Default: `http://127.0.0.1:8082`
* To run predictions on models on a public IP address, specify the IP address as `0.0.0.0`.
  To run predictions on models on a specific IP address, specify the IP address and port.

```properties
# bind inference API to all network interfaces with SSL enabled
inference_address=https://0.0.0.0:8443
```

```properties
# bind inference API to private network interfaces
inference_address=https://172.16.1.10:8080
```

### Configure TorchServe gRPC listening ports
The inference gRPC API is listening on port 7070, and the management gRPC API is listening on port 7071 by default.

To configure different ports use following properties

* `grpc_inference_port`: Inference gRPC API binding port. Default: 7070
* `grpc_management_port`: management gRPC API binding port. Default: 7071

Here are a couple of examples:

### Enable SSL

To enable HTTPs, you can change `inference_address`, `management_address` or `metrics_address` protocol from http to https. For example: `inference_address=https://127.0.0.1`.
The default is port 443, but you can make TorchServe listen on whatever port you set to accept https requests.
For example, to receive https traffic on port 8443, you would use: `inference_address=https://127.0.0.1:8443`.

You must also provide a certificate and private key to enable SSL. TorchServe supports two ways to configure SSL:

1. Use a keystore:
  * **keystore:** the keystore file location. If multiple private key entries exist in the keystore, the first one will be used.
  * **keystore_pass**: the keystore password. The password (if applicable) MUST be the same as keystore password.
  * **keystore_type**: the type of keystore. Default: PKCS12.

2. Use private-key/certificate files:
  * **private_key_file**: the private key file location. Supports both PKCS8 and OpenSSL private keys.
  * **certificate_file**: the X509 certificate chain file location.

#### Examples

**Option 1**: Use a keystore; generate a keystore with Java's keytool. Note the `storepass` argument expects you to create your own password.

```bash
keytool -genkey -keyalg RSA -alias ts -keystore keystore.p12 -storepass changeit -storetype PKCS12 -validity 3600 -keysize 2048 -dname "CN=www.MY_TS.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"
```

Configure the following properties in config.properties:

```bash
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
metrics_address=https://127.0.0.1:8445
keystore=keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
```

**Option 2**: Use private-key/certificate files; generate your self signed cert and key with OpenSSL:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```

Config following property in config.properties:

```properties
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
metrics_address=https://127.0.0.1:8445
private_key_file=mykey.key
certificate_file=mycert.pem
```

### Configure Cross-Origin Resource Sharing (CORS)

CORS is a mechanism that uses additional HTTP headers to tell a browser to let a
web application running at one origin (domain) have permission to access selected
resources from a server at a different origin.

CORS is disabled by default. Configure following properties in config.properties file to enable CORS:

```properties
# cors_allowed_origin is required to enable CORS, use '*' or your domain name
cors_allowed_origin=https://yourdomain.com
# required if you want to use preflight request
cors_allowed_methods=GET, POST, PUT, OPTIONS
# required if the request has an Access-Control-Request-Headers header
cors_allowed_headers=X-Custom-Header
```

### Prefer direct buffer
Configuration parameter prefer_direct_buffer controls if the model server will be using direct memory specified by -XX:MaxDirectMemorySize. This parameter is for model server only and  doesn't affect other packages' usage of direct memory buffer. Default: false

```properties
prefer_direct_buffer=true
```

### Allow model specific custom python packages.
Custom models/handlers may depend on different python packages which are not installed by-default as a part of `TorchServe` setup. User can supply a requirements.txt file containing the required list of python packages to be installed by `TorchServe` for seamless model serving.
Configuration parameter `install_py_dep_per_model` controls if the model server will install the python packages using the `requirements` file supplied with the model archive. Default: false

```properties
install_py_dep_per_model=true
```

User can also supply custom python packages in zip or tar.gz format using the `--extra-files` flag while creating the model-archive and make an entry of the file name in the `requirements` file.

### Restrict backend worker to access environment variables

Environment variables might contain sensitive information, like AWS credentials. Backend workers execute an arbitrary model's custom code,
which might expose a security risk. TorchServe provides a `blacklist_env_vars` property that allows you to restrict which environment variables can be accessed by backend workers.

* `blacklist_env_vars`: a regular expression to filter out environment variable names. Default: all environment variables are visible to backend workers.

### Limit GPU usage

By default, TorchServe uses all available GPUs for inference. Use `number_of_gpu` to limit the usage of GPUs.

* `number_of_gpu`: Maximum number of GPUs that TorchServe can use for inference. Default: all available GPUs in system.

### Nvidia control Visibility

Set nvidia environment variables. For example:

* export CUDA_DEVICE_ORDER="PCI_BUS_ID"
* export CUDA_VISIBLE_DEVICES="1,3"

### Enable metrics api
* `enable_metrics_api` : Enable or disable metric apis i.e. it can be either `true` or `false`. Default: true (Enabled)

### Config model
* `models`: Use this to set configurations specific to a model. The value is presented in json format.
```
{
    "modelName": {
        "version": {
            "parameterName1": parameterValue1,
            "parameterName2": parameterValue2,
            "parameterNameN": parameterValueN,
        }
    }
}
```
A model's parameters are defined in [model source code](https://github.com/pytorch/serve/blob/a9e218ae95fe7690c84b555d0fb9021322c9b049/frontend/archive/src/main/java/org/pytorch/serve/archive/model/ModelConfig.java#L11)


* `minWorkers`: the minimum number of workers of a model
* `maxWorkers`: the maximum number of workers of a model
* `batchSize`: the batch size of a model
* `maxBatchDelay`: the maximum delay in msec of a batch of a model
* `responseTimeout`: the timeout in sec of a specific model's response. This setting takes priority over `default_response_timeout` which is a default timeout over all models
* `defaultVersion`: the default version of a model
* `marName`: the mar file name of a model

A model's configuration example
```properties
models={\
  "noop": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "noop.mar",\
        "minWorkers": 1,\
        "maxWorkers": 1,\
        "batchSize": 4,\
        "maxBatchDelay": 100,\
        "responseTimeout": 120\
    }\
  },\
  "vgg16": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "vgg16.mar",\
        "minWorkers": 1,\
        "maxWorkers": 4,\
        "batchSize": 8,\
        "maxBatchDelay": 100,\
        "responseTimeout": 120\
    }\
  }\
}
```
Starting from version 0.8.0, TorchServe allows for model configuration using a YAML file embedded in the MAR file. This YAML file contains two distinct parts that determine how a model is configured: frontend parameters and backend parameters. (see [details](https://github.com/pytorch/serve/tree/master/model-archiver#config-file))

* The frontend parameters are controlled by TorchServe's frontend and specify the parameter name and default values. TorchServe now uses a priority order to determine the final value of a model's parameters in frontend. Specifically, the config.property file has the lowest priority, followed by the model configuration YAML file, and finally, the REST or gRPC model management API has the highest priority.

* The backend parameters are fully controlled by the user. Users customized handler can access the backend parameters via the `model_yaml_config` property of the [context object](https://github.com/pytorch/serve/blob/a9e218ae95fe7690c84b555d0fb9021322c9b049/ts/context.py#L24). For example, context.model_yaml_config["pippy"]["rpc_timeout"].

* User can allocate specific GPU device IDs to a model by defining "deviceIds" in the frontend parameters in the YAML file. TorchServe uses a round-robin strategy to assign device IDs to a model's worker. If specified in the YAML file, it round-robins the device IDs listed; otherwise, it uses all visible device IDs on the host.

### Other properties

Most of the following properties are designed for performance tuning. Adjusting these numbers will impact scalability and throughput.

* `enable_envvars_config`: Enable configuring TorchServe through environment variables. When this option is set to "true", all the static configurations of TorchServe can come through environment variables as well. Default: false
* `number_of_netty_threads`: Number frontend netty thread. This specifies the number of threads in the child [EventLoopGroup](https://livebook.manning.com/book/netty-in-action/chapter-8) of the frontend netty server. This group provides EventLoops for processing Netty Channel events (namely inference and management requests) from accepted connections. Default: number of logical processors available to the JVM.
* `netty_client_threads`: Number of backend netty thread. This specifies the number of threads in the WorkerThread [EventLoopGroup](https://livebook.manning.com/book/netty-in-action/chapter-8) which writes inference responses to the frontend. Default: number of logical processors available to the JVM.
* `default_workers_per_model`: Number of workers to create for each model that loaded at startup time. Default: available GPUs in system or number of logical processors available to the JVM.
* `job_queue_size`: Number inference jobs that frontend will queue before backend can serve. Default: 100.
* `async_logging`: Enable asynchronous logging for higher throughput, log output may be delayed if this is enabled. Default: false.
* `default_response_timeout`: Timeout, in seconds, used for all models backend workers before they are deemed unresponsive and rebooted. Default: 120 seconds.
* `unregister_model_timeout`: Timeout, in seconds, used when handling an unregister model request when cleaning a process before it is deemed unresponsive and an error response is sent. Default: 120 seconds.
* `decode_input_request`: Configuration to let backend workers to decode requests, when the content type is known.
If this is set to "true", backend workers do "Bytearray to JSON object" conversion when the content type is "application/json" and
the backend workers convert "Bytearray to utf-8 string" when the Content-Type of the request is set to "text*". Default: true
* `initial_worker_port` : This is the initial port number for auto assigning port to worker process.
* `model_store` : Path of model store directory.
* `model_server_home` : Torchserve home directory.
* `max_request_size` : The maximum allowable request size that the Torchserve accepts, in bytes. Default: 6553500
* `max_response_size` : The maximum allowable response size that the Torchserve sends, in bytes. Default: 6553500
* `limit_max_image_pixels` : Default value is true (Use default [PIL.Image.MAX_IMAGE_PIXELS](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.MAX_IMAGE_PIXELS)). If this is set to "false", set PIL.Image.MAX_IMAGE_PIXELS = None in backend default vision handler for large image payload.
* `allowed_urls` : Comma separated regex of allowed source URL(s) from where models can be registered. Default: `file://.*|http(s)?://.*` (all URLs and local file system)
e.g. : To allow base URLs `https://s3.amazonaws.com/` and `https://torchserve.pytorch.org/` use the following regex string `allowed_urls=https://s3.amazonaws.com/.*,https://torchserve.pytorch.org/.*`
* `workflow_store` : Path of workflow store directory. Defaults to model store directory.
* `disable_system_metrics` : Disable collection of system metrics when set to "true". Default value is "false".

**NOTE**

All the above config properties can be set using environment variable as follows.
- set `enable_envvars_config` to true in config.properties
- export environment variable for property as`TS_<PROPERTY_NAME>`.

  e.g.: to set inference_address property run cmd
  `export TS_INFERENCE_ADDRESS="http://127.0.0.1:8082"`.

---
