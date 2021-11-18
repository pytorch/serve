# Serve a MNIST Model for Inference with KServe

In this document, the .mar file creation, request & response on the KServe side and the KServe changes to the handler files for Image Classification model using Torchserve's default vision handler.

## .mar file creation

The .mar file creation command is as below:
```bash
torch-model-archiver --model-name mnist --version 1.0 --model-file serve/examples/image_classifier/mnist/mnist.py --serialized-file serve/examples/image_classifier/mnist/mnist_cnn.pt --handler  serve/examples/image_classifier/mnist/mnist_handler.py
```

## Starting Torchserve 
To serve an Inference Request for Torchserve using the KServe Spec, follow the below:

* create a config.properties file and specify the details as shown:

```
inference_address=http://127.0.0.0:8085
management_address=http://127.0.0.0:8081
number_of_netty_threads=4
enable_envvars_config=true
job_queue_size=10
model_store=model_store
```

* Set service envelope environment variable

The
`export TS_SERVICE_ENVELOPE=kserve` or `export TS_SERVICE_ENVELOPE=kservev2` envvar is for choosing between
KServe v1 and v2 protocols. This is set by the controller in KServe cluster.

* start Torchserve by invoking the below command:

```bash
torchserve --start --model-store model_store --ncs --models mnist=mnist.mar
```

## Register model

Hit the below curl request to register the model

```bash
curl -X POST "localhost:8081/models?model_name=mnist&url=mnist.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```
Please note that the batch size, the initial worker and synchronous values can be changed at your discretion and they are optional.

## Request and Response

### The curl request for Inference is as below:

The image file can be converted into string of bytes array by running
  
```bash
python img2bytearray.py <imagefile>
```

When the curl request is made, ensure that the request is made inisde of the serve folder.
```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/mnist.json http://127.0.0.1:8085/v1/models/mnist:predict
```
The default Inference Port for Torchserve is 8080, while for KServe it is 8085

The Prediction response is as below :

```json
{
  "predictions": [
    2
  ]
}
```


### The curl request for Explanation is as below:

Torchserve supports KServe Captum Explanations for Eager Models only.

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/mnist.json http://127.0.0.1:8085/v1/models/mnist:explain
```

The Explanation response is as below :

```json
{
  "explanations": [
    [
      [
        [
          0.004570948731989492,
          0.006216969640322402,
          0.008197565423679522,
          0.009563574612830427,
          0.008999274832810742,
          0.009673474804303854,
          0.007599905146155397,
          ,
          ,

        ]
      ]
    ]
  ]
}
```

### Static batching:

KServe supports Static batching by adding new examples in the instances key of the request json
But the batch size should still be set at 1, when we register the model. Explain doesn't support batching.


```json
{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    },
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
  ]
}
```

### The curl request for the Server Health check 

Server Health check API returns the model's state for inference

```bash
curl -X GET "http://127.0.0.1:8081/v1/models/mnist"
```

The response is as below:

```json
{
  "name": "mnist",
  "ready": true
}
```

## The KServe Changes in the handler files


* 1)  When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.
The mnist request difference between the regular torchserve and kserve is as below

	### Regular torchserve request

	```json
	[
			{

		"data" :  "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"

			}

	]	
	```

	### KServe Request:

	```json
	{
	"instances": [
		{
			"data": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
		}
	]
	}
	```

	The KServe request is unwrapped by the kserve envelope in torchserve  and sent like a torchserve request. So effectively the values of  `instances`  key is sent to the handlers.


* 2) Torchserve handles the input request for Image Classification tasks in the format of BytesArray. On the KServe side, the predictor does not take the request as bytesarray (Image Transformer Functionality in KServe converts the BytesArray into a JSON array) for details refer the Image Transformer section(step 5 and step 10) in the [End to End Transformer](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/README.md). The code change is done to handle both BytesArray and JSON array Input Requests as a part of the pre-process method of [vision_handler.py](https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py).
