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



#Serve a custom model on Torchserve with KFServing API Spec for Inference:



To serve the model in KFserving for Inference, follow the below steps :

* Step 1 : specify kfserving as the envelope in the config.properties file as below :

'''
service_envelope=kfserving
'''

* Step 1 : Create a .mar file by invoking the below command :

'''
torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file serve/examples/text_classification/model.py --serialized-file serve/examples/text_classification/model.pt --handler text_classifier --extra-files "serve/examples/text_classification/index_to_name.json,serve/examples/text_classification/source_vocab.pt"
'''

* Step - 2 : Ensure that the docker image for Torchserve is created and accessible by the KFServing Environment. 
	     Refer the document for creating torchserve image with kfserving wrapper 

* Step - 3 : Create an Inference Service in the Kubeflow, refer to the doc below to initiate the process:
<the doc link>

* Step - 4 : Make a postman request as below :

Name of the POST URL  : <inference request address>/v1/models/my_tc:predict

* Step - 5	 : In the request body, specify the request as below:

{
"instances":[{
            "name":"context",
            "data":"Bloomberg has reported on the economy"

  }]    
}


.

The Prediction response is as below :

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





