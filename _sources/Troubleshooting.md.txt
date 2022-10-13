## Troubleshooting Guide
Refer to this section for common issues faced while deploying your Pytorch models using Torchserve and their corresponding troubleshooting steps.

* [Deployment and config issues](#deployment-and-config-issues)
* [Snapshot related issues](#snapshot-related-issues)
* [API related issues](#api-relate-issues)
* [Model-archiver](#model-archiver)


### Deployment and config issues
#### "Failed to bind to address: `http://127.0.0.1:8080`", port 8080/8081 already in use.
Usually, the port number 8080/8081 is already used by some other application or service, it can be verified by using cmd `ss -ntl | grep 8080`. There are two ways to troubleshoot this issue either kill the process which is using port 8080/8081 or run Torchserve on different ports other than 8080 & 8081.

Refer [configuration.md](https://github.com/pytorch/serve/blob/master/docs/configuration.md) for more details.

Relevant issues: [[542](https://github.com/pytorch/serve/issues/542)]

#### "java.lang.NoSuchMethodError" when starting Torchserve.[[473](https://github.com/pytorch/serve/issues/473)]
This error usually occurs when Java 17 is not installed or used. Java 17 is required by Torchserve and older java versions are not supported.

Relevant issues: [[#473](https://github.com/pytorch/serve/issues/473)]

####  Unable to send big files for inference request?
The default max request size and response size is roughly 6.5 Mb. Hence any file size greater than 6.5mb cannot be uploaded.
To resolve this update `max_request_size` and `max_response_size` in a config.properties file and start torchserve with this config file.
```
$ cat config.properties
max_request_size=<request size in bytes>
max_response_size=<response size in bytes>
$ torchserve --start --model-store model_store --ts-config /path/to/config.properties
```
You can also use environment variables to set these values.
Refer [configuration.md](https://github.com/pytorch/serve/blob/master/docs/configuration.md) for more details.
Relevant issues: [[#335](https://github.com/pytorch/serve/issues/335)]

###  Snapshot related issues
#### How to disable Snapshot feature?
By default, the snapshot feature is enabled. To disable snapshot feature start torchserve using --ncs flag or specify config file using --ts-config path/to/config
Relevant issues:[[#383](https://github.com/pytorch/serve/issues/383), [#512](https://github.com/pytorch/serve/issues/512)]

#### Torchserve stopped after restart with "InvalidSnapshotException" exception.
Torchserve when restarted uses the last snapshot config file to restore its state of models and their number of workers. When "InvalidSnapshotException" is thrown then the model store is in an inconsistent state as compared with the snapshot. To resolve this the snapshot config files can be removed or torchserve can be started with specific config file using --ts-config path/to/config.
Refer [snapshot.md](https://github.com/pytorch/serve/blob/master/docs/snapshot.md) for more details.

####  Where are snapshot config files stored?
The snapshots are by default in `{LOG_LOCATION}\config` directory, where `{LOG_LOCATION}` is a system environment variable that can be used by TorchServe. If this variable is not set, the snapshot is stored in `.\log\config` directory
Refer [snapshot.md](https://github.com/pytorch/serve/blob/master/docs/snapshot.md) for more details.

#### How to change Temp Directory?
You can export TEMP environment variable to the desired path. This path will be used by TorchServe to extract the model-archives content.
Relevant issues: [[#654](https://github.com/pytorch/serve/issues/654)]

###  API related issues

#### Register model: Failed with exception "ConflictStatusException" & error code 409.
This gives a clear message that the model we are trying to register conflicts with an already existing model. To resolve this change the model version when creating a mar file or register a model with a different name.
Relevant issues: [[#500](https://github.com/pytorch/serve/issues/500)]

#### Register model: Failed with exception "DownloadModelException" & error code 400.
Torchserve was unable to download the mar file in this case. To resolve this check whether the given URL is accessible publicly.

#### Register model: Failed with exception "ModelNotFoundException" & error code 404.
In this case, Torchserve was unable to locate a given mar file in the model store directory. To resolve this check whether the given mar file exists in the model store. Check the mar file name in the POST request to register the model.

#### Inference request: Failed with exception "ServiceUnavailableException" & error code 503.
In this case, the model is registered but there no workers spawned for the model. Use the scale-up API to increase the number of workers. You can verify the number of workers using
`curl -X GET"http://localhost:8081/models/<model_name>"
`
### Model-archiver

#### How can add model  specific custom dependency?
You can add your dependency files using `--extra-files` flag while creating a mar file. These dependency files can be of any type like zip, egg, json etc. You may have to write a custom handler to use these files as required.

Relevant issues: [[#566](https://github.com/pytorch/serve/issues/566)]

#### How can I resolve model specific python dependency?
You can provide a requirements.txt while creating a mar file using "--requirements-file/ -r" flag. You can refer to the [waveglow text-to-speech-synthesizer](https://github.com/pytorch/serve/tree/master/examples/text_to_speech_synthesizer) example

-   [waveglow mar creation script](https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/create_mar.sh)
-   [waveglow handler](https://github.com/pytorch/serve/blob/2d9c7ccc316f592374943a1963c1057bbe232c9e/examples/text_to_speech_synthesizer/waveglow_handler.py#L49)

Relevant issues: [[#566](https://github.com/pytorch/serve/issues/566)]
Refer [Torch model archiver cli](https://github.com/pytorch/serve/blob/master/model-archiver/README.md#torch-model-archiver-command-line-interface) for more details.

#### I have added  requirements.txt in my mar file but the packages listed are not getting installed.
By default model specific custom python packages feature is disabled, enable this by setting install_py_dep_per_model to true.
Refer [Allow model specific custom python packages](https://github.com/pytorch/serve/blob/master/docs/configuration.md#allow-model-specific-custom-python-packages) for more details.


#### Backend worker monitoring thread interrupted or backend worker process died error.
This issue mostly occurs when the model fails to initialize, which may be due to erroneous code in handler's initialize function. This error is also observed when there is missing package/module.

Relevant issues: [[#667](https://github.com/pytorch/serve/issues/667), [#537](https://github.com/pytorch/serve/issues/537)]
