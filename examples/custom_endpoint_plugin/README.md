# Torchserve custom endpoint plugin

In this example, we demonstrate how to create a custom HTTP API endpoint plugin for TorchServe. Endpoint plugins enable us to dynamically add custom functionality to TorchServe at start time, without having to rebuild TorchServe. For more details on endpoint plugins and TorchServe SDK, refer to the following links:
- [Plugins Readme](https://github.com/pytorch/serve/tree/master/plugins)
- [TorchServe SDK source](https://github.com/pytorch/serve/tree/master/serving-sdk)

In this example, we will build an endpoint plugin that implements the functionality of a HTTP API endpoint that reports the readiness of models registered on TorchServe to serve inference requests.

Run the commands given in the following steps from the root directory of the repository. For example, if you cloned the repository into `/home/my_path/serve`, run the steps from `/home/my_path/serve`

## Steps

- Step 1: Install the necessary dependencies for TorchServe development environment

  ```bash
  $ python ts_scripts/install_dependencies.py --environment=dev
  ```

- Step 2: Copy [ModelReady.java](ModelReady.java) to the endpoint plugins directory

  ```bash
  $ cp examples/custom_endpoint_plugin/ModelReady.java plugins/endpoints/src/main/java/org/pytorch/serve/plugins/endpoint
  ```
  For reference on implemeting your own custom plugin, review the utilization of the [TorchServe SDK API](https://github.com/pytorch/serve/tree/master/serving-sdk/src/main/java/org/pytorch/serve/servingsdk) and its corresponding [implementation](https://github.com/pytorch/serve/tree/master/frontend/server/src/main/java/org/pytorch/serve/servingsdk/impl) in [ModelReady.java](ModelReady.java).

- Step 3: Copy [org.pytorch.serve.servingsdk.ModelServerEndpoint](org.pytorch.serve.servingsdk.ModelServerEndpoint) to the plugins service provider configuration directory

  ```bash
  $ cp examples/custom_endpoint_plugin/org.pytorch.serve.servingsdk.ModelServerEndpoint plugins/endpoints/src/main/resources/META-INF/services
  ```

- Step 4: Update the [endpoint plugins build script](../../plugins/endpoints/build.gradle) to only include the required plugins in the JAR

  ```bash
  .....
  .....
  /**
   * By default, include all endpoint plugins in the JAR.
   * In order to build a custom JAR with specific endpoint plugins, specify the required paths.
   * For example:
   * include "org/pytorch/serve/plugins/endpoint/Ping*"
   * include "org/pytorch/serve/plugins/endpoint/ExecutionParameters*"
   */
  include "org/pytorch/serve/plugins/endpoint/ModelReady*"
  .....
  .....
  ```

- Step 5: Build the custom endpoint plugin

  ```bash
  $ cd plugins
  $ ./gradlew clean build
  $ cd ..
  ``` 

- Step 6: Create two example model archives to test the plugin with

  ```bash
  $ mkdir -p model_store
  $ torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
  $ mv mnist.mar ./model_store 
  ```

  ```bash 
  $ wget https://download.pytorch.org/models/resnet18-f37072fd.pth
  $ torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json
  $ mv resnet-18.mar ./model_store
  ```

- Step 7: Start Torchserve with the appropriate plugins path containing the JAR we just built.
  The plugin JAR will be contained in the `plugins/endpoints/build/libs` directory. For Ex: `plugins/endpoints/build/libs/endpoints-1.0.jar`
  ```bash
  $ torchserve --ncs --start --model-store ./model_store --disable-token-auth --enable-model-api --plugins-path ./plugins/endpoints/build/libs
  ```

- Step 8: Register the models and test the custom endpoint

  ```bash
  $ curl -X POST  "http://localhost:8081/models?url=mnist.mar"
  {
    "status": "Model \"mnist\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
  }

  $ curl -X POST  "http://localhost:8081/models?url=resnet-18.mar"
  {
    "status": "Model \"resnet-18\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
  }
  ```

  ```bash
  $ curl -X GET http://localhost:8080/model-ready
  {
    "Status": "Model/s not ready"
  }
  ```

  The `model-ready` endpoint reports that the models are not ready since there are no workers that have loaded the models and ready to serve inference requests.

- Step 9: Scale up workers for one of the models and test the custom endpoint
  
  ```bash
  $ curl -X PUT "http://localhost:8081/models/mnist?min_worker=1&synchronous=true"
  {
    "status": "Workers scaled to 1 for model: mnist"
  }
  ```

  ```bash
  $ curl -X GET http://localhost:8080/model-ready
  {
    "Status": "Model/s not ready"
  }
  ```

  The `model-ready` endpoint reports that the models are not ready since not all registered models have atleast one worker ready to serve inference requests.

- Step 10: Scale up workers for both models and test the custom endpoint

  ```bash
  $ curl -X PUT "http://localhost:8081/models/mnist?min_worker=1&synchronous=true"
  {
    "status": "Workers scaled to 1 for model: mnist"
  }

  $ curl -X PUT "http://localhost:8081/models/resnet-18?min_worker=1&synchronous=true"
  {
    "status": "Workers scaled to 1 for model: resnet-18"
  }
  ```

  ```bash
  $ curl -X GET http://localhost:8080/model-ready
  {
    "Status": "Model/s ready"
  }
  ```

  The `model-ready` endpoint reports that the models are now ready since there is atleast one worker per registered model that is ready to serve inference requests.

- Step 11: Stop TorchServe
  ```bash
  $ torchserve --stop
  ```

