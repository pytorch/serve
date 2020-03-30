

## Serve a Model

After you install TorchServe, you can get a TorchServe model server up and running very quickly. Try out `--help` to see all the CLI options available.

```bash
torchserve --help
```

Now, we'll  [full server docs](docs/server.md) when you're ready.

Here is an easy example for serving an object classification model:

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv densenet161.mar model_store/
torchserve --start --model-store model_store --models densenet161=densenet161.mar
```

With the command above executed, you have TorchServe running on your host, listening for inference requests. **Please note, that if you specify model(s) during TorchServe start - it will automatically scale backend workers to the number equal to available vCPUs (if you run on CPU instance) or to the number of available GPUs (if you run on GPU instance). In case of powerful hosts with a lot of compute resoures (vCPUs or GPUs) this start up and autoscaling process might take considerable time. If you would like to minimize TorchServe start up time you can try to avoid registering and scaling up model during start up time and move that to a later point by using corresponding [Management API](docs/management_api.md#register-a-model) calls (this allows finer grain control to how much resources are allocated for any particular model).**

To test it out, you can open a new terminal window next to the one running TorchServe. Then you can use `curl` to download one of these [cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images) and curl's `-o` flag will name it `kitten.jpg` for you. Then you will `curl` a `POST` to the TorchServe predict endpoint with the kitten's image.

![kitten](docs/images/kitten_small.jpg)

In the example below, we provide a shortcut for these steps.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

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

## Stopping the running TorchServe

To stop the current running TorchServe instance, run the following command:

```bash
torchserve --stop
```

You would see output specifying that TorchServe has stopped.

## Create a Model Archive

TorchServe enables you to package up all of your model artifacts into a single model archive. This makes it easy to share and deploy your models.
To package a model, check out [model archiver documentation](model-archiver/README.md)

## Recommended production deployments

* TorchServe doesn't provide authentication. You have to have your own authentication proxy in front of TorchServe.
* TorchServe doesn't provide throttling, it's vulnerable to DDoS attack. It's recommended to running TorchServe behind a firewall.
* TorchServe only allows localhost access by default, see [Network configuration](docs/configuration.md#configure-ts-listening-port) for detail.
* SSL is not enabled by default, see [Enable SSL](docs/configuration.md#enable-ssl) for detail.
* TorchServe use a config.properties file to configure TorchServe's behavior, see [Manage TorchServe](docs/configuration.md) page for detail of how to configure TorchServe.