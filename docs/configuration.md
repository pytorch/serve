# Advanced configuration

One of design goal of TS is easy to use. The default settings form TS should be sufficient for most of use cases. This document describe advanced configurations that allows user to deep customize TS's behavior.

## Environment variables

User can set environment variables to change TS behavior, following is a list of variables that user can set for TS:
* JAVA_HOME
* PYTHONPATH
* TS_CONFIG_FILE
* LOG_LOCATION
* METRICS_LOCATION

**Note:** environment variable has higher priority that command line or config.properties. It will override other property values.

## Command line parameters

User can following parameters to start TS, those parameters will override default TS behavior:

* **--ts-config** TS will load specified configuration file if TS_CONFIG_FILE is not set.
* **--model-store** This parameter will override `model_store` property in config.properties file.
* **--models** This parameter will override `load_models' property in config.properties.
* **--log-config** This parameter will override default log4j.properties.
* **--foreground** This parameter will run the model server in foreground. If this option is
                        disabled, the model server will run in the background.

See [Running the Model Server](server.md) for detail.

## config.properties file

TS use a `config.properties` file to store configurations. TS use following order to locate this `config.properties` file:
1. if `TS_CONFIG_FILE` environment variable is set, TS will load the configuration from the environment variable.
2. if `--ts-config` parameter is passed to `torchserve`, TS will load the configuration from the parameter.
3. if there is a `config.properties` in current folder where user start the `torchserve`, TS will load the `config.properties` file form current working directory.
4. If none of above is specified, TS will load built-in configuration with default values.

**Note:** Docker image that TS provided has slightly different default value.

### Customize JVM options

The restrict TS frontend memory footprint, certain JVM options is set via **vmargs** property in `config.properties` file

* default: N/A, use JVM default options
* docker default: -Xmx128m -XX:-UseLargePages -XX:+UseG1GC -XX:MaxMetaspaceSize=32M -XX:MaxDirectMemorySize=10m -XX:OnOutOfMemoryError='kill -9 %p'

User can adjust those JVM options for fit their memory requirement if needed.

### Load models at startup

User can configure load models while TS startup. TS can load models from `model_store` or from HTTP(s) URL.

* model_store
	* standalone: default: N/A, load models from local disk is disabled.
	* docker: default model_store location is set to: /opt/ml/model

* load_models
	* standalone: default: N/A, no models will be load on startup.
	* docker: default: ALL, all model archives in /opt/ml/model will be loadded on startup.

**Note:** `model_store` and `load_models` property can be override by command line parameters.

### Configure TS listening port

TS doesn't support authentication natively. To avoid unauthorized access, TS only allows localhost access by default. Inference API is listening on 8080 port and accepting HTTP request. Management API is listening on 8081 port and accepting HTTP request. See [Enable SSL](#enable-ssl) for configuring HTTPS.

* inference_address: inference API binding address, default: http://127.0.0.1:8080
* management_address: management API binding address, default: http://127.0.0.1:8081

Here are a couple of examples:
```properties
# bind inference API to all network interfaces with SSL enabled
inference_address=https://0.0.0.0:8443

# bind inference API to private network interfaces
inference_address=https://172.16.1.10:8080
```

### Enable SSL

For users who want to enable HTTPs, you can change `inference_address` or `management_addrss` protocol from http to https, for example: `inference_addrss=https://127.0.0.1`. This will make TS listening on localhost 443 port to accepting https request.

User also must provide certificate and private keys to enable SSL. TS support two ways to configure SSL:
1. Use keystore
	* keystore: Keystore file location, if multiple private key entry in the keystore, first one will be picked. 
	* keystore_pass: keystore password, key password (if applicable) MUST be the same as keystore password.
    * keystore_type: type of keystore, default: PKCS12

2. Use private-key/certificate files
	* private_key_file: private key file location, support both PKCS8 and OpenSSL private key.
	* certificate_file: X509 certificate chain file location.

#### Self-signed certificate example

This is a quick example to enable SSL with self-signed certificate

1. User java keytool to create keystore
```bash
keytool -genkey -keyalg RSA -alias ts -keystore keystore.p12 -storepass changeit -storetype PKCS12 -validity 3600 -keysize 2048 -dname "CN=www.MY_TS.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"
```

Config following property in config.properties:

```properties
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
keystore=keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
```

2. User OpenSSL to create private key and certificate
```bash
```


Config following property in config.properties:

```properties
inference_address=https://127.0.0.1:8443
management_address=https://127.0.0.1:8444
keystore=keystore.p12
keystore_pass=changeit
keystore_type=PKCS12
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


### Restrict backend worker to access environment variable

Environment variable may contains sensitive information like AWS credentials. Backend worker will execute arbitrary model's custom code, which may expose security risk. TS provides a `blacklist_env_vars` property which allows user to restrict which environment variable can be accessed by backend worker.

* blacklist_env_vars: a regular expression to filter out environment variable names, default: all environment variable will be visible to backend worker.

### Limit GPU usage
By default, TS will use all available GPUs for inference, you use `number_of_gpu` to limit the usage of GPUs.

* number_of_gpu: max number of GPUs that TS can use for inference, default: available GPUs in system.

### Other properties

Most of those properties are designed for performance tuning. Adjusting those numbers will impact scalability and throughput.

* enable_envvars_config: Enable configuring TS through environment variables. When this option is set to "true", all the static configurations of TS can come through environment variables as well. default: false
* number_of_netty_threads: number frontend netty thread, default: number of logical processors available to the JVM.
* netty_client_threads: number of backend netty thread, default: number of logical processors available to the JVM.
* default_workers_per_model: number of workers to create for each model that loaded at startup time, default: available GPUs in system or number of logical processors available to the JVM.
* job_queue_size: number inference jobs that frontend will queue before backend can serve, default 100.
* async_logging: enable asynchronous logging for higher throughput, log output may be delayed if this is enabled, default: false.
* default_response_timeout: Timeout, in seconds, used for model's backend workers before they are deemed unresponsive and rebooted. default: 120 seconds.
* unregister_model_timeout: Timeout, in seconds, used when handling an unregister model request when cleaning a process before it is deemed unresponsive and an error response is sent. default: 120 seconds.
* decode_input_request: Configuration to let backend workers to decode requests, when the content type is known. 
If this is set to "true", backend workers do "Bytearray to JSON object" conversion when the content type is "application/json" and 
the backend workers convert "Bytearray to utf-8 string" when the Content-Type of the request is set to "text*". default: true  

### config.properties Example

See [config.properties for docker](https://github.com/pytorch/serve/blob/master/docker/config.properties)

