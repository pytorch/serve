
# Quick start

This topic shows a simple example of serving a model with TorchServe. To complete this example, you must have already installed TorchServe and the model archiver. 
For installation instructions, see the [TorchServe README](../README.md)

## Store a Model

To serve a model with TorchServe, first archive the model as a MAR file. You can use the model archiver to package a model.
You can also create model stores to store your archived models.

The following code gets a trained model, archives the model by using the model archiver, and then stores the model in a model store.

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv densenet161.mar model_store/
```

For more information about the model archiver, see [Torch Model archiver for TorchServe](../model-archiver/README.md)

## Serve a Model

After you archive and store the model, use the `torchserve` command to serve the model.

```bash
torchserve --start --model-store model_store --models densenet161=densenet161.mar
```

After you execute the `torchserve` command above, TorchServe runs on your host, listening for inference requests.

**Note**: If you specify model(s) when you run TorchServe, it automatically scales backend workers to the number equal to available vCPUs (if you run on a CPU instance) or to the number of available GPUs (if you run on a GPU instance). In case of powerful hosts with a lot of compute resoures (vCPUs or GPUs). This start up and autoscaling process might take considerable time. If you want to minimize TorchServe start up time you avoid registering and scaling the model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model), which allows finer grain control of the resources that are allocated for any particular model).

## Get predictions from a model

To test the model server, send a request to the server's `predictions` API.

Comlete the following steps:
* Open a new terminal window (other than the one running TorchServe).
* Use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images)
  and use the  `-o` flag to name it `kitten.jpg` for you.
* Use `curl` to send `POST` to the TorchServe `predict` endpoint with the kitten's image.

![kitten](docs/images/kitten_small.jpg)

The following code completes all three steps:

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

The predict endpoint returns a prediction response in JSON. It will look something like the following result:

```json
[
  {
    "tiger_cat": 0.46933549642562866
  },
  {
    "tabby": 0.4633878469467163
  },
  {
    "Egyptian_cat": 0.06456148624420166
  },
  {
    "lynx": 0.0012828214094042778
  },
  {
    "plastic_bag": 0.00023323034110944718
  }
]
```

You will see this result in the response to your `curl` call to the predict endpoint, and in the server logs in the terminal window running TorchServe. It's also being [logged locally with metrics](docs/metrics.md).

Now you've seen how easy it can be to serve a deep learning model with TorchServe! [Would you like to know more?](docs/server.md)

## Stop the running TorchServe

To stop the currently running TorchServe instance, run the following command:

```bash
torchserve --stop
```

You see output specifying that TorchServe has stopped.

## Recommended production deployments

* TorchServe doesn't provide authentication. You have to have your own authentication proxy in front of TorchServe.
* TorchServe doesn't provide throttling, it's vulnerable to DDoS attack. It's recommended to running TorchServe behind a firewall.
* TorchServe only allows localhost access by default, see [Network configuration](docs/configuration.md#configure-ts-listening-port) for detail.
* SSL is not enabled by default, see [Enable SSL](docs/configuration.md#enable-ssl) for detail.
* TorchServe use a config.properties file to configure TorchServe's behavior, see [Manage TorchServe](docs/configuration.md) page for detail of how to configure TorchServe.