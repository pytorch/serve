# Text Classfication using TorchServe's default text_classifier handler

This is an example to create a text classification dataset and train a sentiment model. We have used the following torchtext example to train the model.

https://github.com/pytorch/text/tree/master/examples/text_classification

We have copied the files from above example and made small changes to save the model's state dict and added default values.

# Training the model

Run the following commands to train the model :

```bash
./run_script.sh
```

The above command generated the model's state dict as model.pt and the vocab used during model training as source_vocab.pt

# Serve the text classification model on TorchServe

 * Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file model.py --serialized-file model.pt  --handler text_classifier --extra-files "index_to_name.json,source_vocab.pt"
    ```
    
    NOTE - `run_script.sh` has generated `source_vocab.pt` and it is a mandatory file for this handler. 
           If you are planning to override or use custom source vocab. then name it as `source_vocab.pt` and provide it as `--extra-files` as per above example.
           Other option is to extend `TextHandler` and override `get_source_vocab_path` function in your custom handler. Refer [custom handler](../../docs/custom_service.md) for detail
   
 * Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv my_text_classifier.mar model_store/
    torchserve --start --model-store model_store --models my_tc=my_text_classifier.mar
    curl http://127.0.0.1:8080/predictions/my_tc -T examples/text_classification/sample_text.txt
    ```
To make a captum explanations request on the Torchserve side, use the below command:

```bash
curl -X POST http://127.0.0.1:8080/explanations/my_tc -T examples/text_classification/sample_text.txt
```

In order to run Captum Explanations with the request input in a json file, follow the below steps:

In the config.properties, specify `service_envelope=body` and make the curl request as below:
```bash
curl -H "Content-Type: application/json" --data @examples/text_classification/text_classifier_ts.json http://127.0.0.1:8085/explanations/my_tc_explain
```

#Serve a custom model on Torchserve with KFServing API Spec for Inference:



To serve the model in KFserving for Inference, follow the below steps :

* Step 1 : specify kfserving as the envelope in the config.properties file as below :

```bash
service_envelope=kfserving
```

* Step 2 : Create a .mar file by invoking the below command :

```bash
torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file serve/examples/text_classification/model.py --serialized-file serve/examples/text_classification/model.pt --handler text_classifier --extra-files "serve/examples/text_classification/index_to_name.json,serve/examples/text_classification/source_vocab.pt"
```

* Step 3 : Ensure that the docker image for Torchserve is created and accessible by the KFServing Environment. 
	    Refer the document for creating torchserve image with kfserving wrapper
	   

* Step 4 : Create an Inference Service in the Kubeflow, refer to the doc below to initiate the process:
[End to End Torchserve KFServing Model Serving](https://github.com/pytorch/serve/blob/master/kf_predictor_docker/README.md)

* Step 5 : Make the curl request as below for predict:

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kf_request_json/text_classifier_kf.json http://127.0.0.1:8085/v1/models/my_tc:predict
```
.

The Prediction response is as below :

```bash
{
	"predictions" : [
	   {
		"World" : 0.0036
		"Sports" : 0.0065
		"Business" :0.9160
		"Sci/Tec" :0.079
	   }
	]
}
```

* Step 6 : Make the curl request as below for explain :

Make the curl request as below:

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kf_request_json/text_classifier_kf.json http://127.0.0.1:8085/v1/models/my_tc:explain
```

The explanation response is as below :

```bash
{
  "explanations": [
    {
      "importances": [
        [
          0.00017786371265739233,
          0.9824386919377469,
          4.193646962600815e-06,
          0.00014836451651210265,
          6.149280398342056e-05,
          -------,
	  -------
        ]
      ],
      "words": [
        "bloomberg",
        "has",
        "reported",
        "on",
        "the",
        "economy"
      ]
    }
  ]
}
```
