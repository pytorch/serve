# Serve a MNIST Model for Inference on the KFServing side:

In this document, the .mar file creation, request & response on the KFServing side and the KFServing changes to the handler files for Image Classification model using Torchserve's default vision handler.

## .mar file creation

The .mar file creation command is as below:
```bash
torch-model-archiver --model-name mnist --version 1.0 --model-file serve/examples/image_classifier/mnist/mnist.py --serialized-file serve/examples/image_classifier/mnist/mnist_cnn.pt --handler  serve/examples/image_classifier/mnist/mnist_handler.py
```

## Request and Response

The curl request is as below:

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/mnist.json http://127.0.0.1:8085/v1/models/mnist:predict
```

The Prediction response is as below :

```bash
{
	"predictions" : [

						2
					]
}
```


## KFServing changes to the handler files 

* 1)  When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.
The bert request difference between the regular torchserve and kfserving is as below:

	### Regular torchserve request

	```bash
	[
			{

		"data" :  "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"

			}

	  ]	
	```

	### KFServing Request:
	```bash
	{
		"instances":[
						{
						"data" : "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
						}
					]
	}
	```

	The KFServing request is unwrapped by the kfserving envelope in torchserve  and sent like a torchserve request. So effectively the values of  `instances`  key is sent to the handlers.


* 2) Torchserve handles the input request for Image Classification tasks in the format of BytesArray. On the KFServing side, the predictor does not take the request as bytesarray (Image Transformer Functionality in KFServing converts the BytesArray into a JSON array) for details refer the Image Transformer section(step 5 and step 10) in the [End to End Transformer](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/README.md). The code change is done to handle both BytesArray and JSON array Input Requests as a part of the pre-process method of [vision_handler.py](https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py).
