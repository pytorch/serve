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

#Serve the Text classification model with KFServing API spec:

The Inference Request to be hit should follow the below KFServing API Spec:
http://127.0.0.1:8080/v1/models/<modelname>:predict

We have created a handler filed called text_classifier_kf.py and made it as a default handler to serve Text Classification models on the KFServing side. 

We need to add the kfserving as a part of the config.properties file. Place the config.properties in the parent folder where you serve the model from. The content of the config.properties file is as below:

'''
service_envelope=kfserving
'''

The following steps are to be followed to serve the models on the KFServing side:

 * Step - 1: Run the command below to train the model and generate  model.pt and source_vocab.pt files for .mar file creation. Point the directory to “serve/examples/text_classification” and open a terminal and write the command below:
	'''
	./run_script.sh

	'''

 * Step - 2 : Create the model archive file for the text classification problem. 
	'''
	torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file serve/examples/text_classification/model.py --serialized-file serve/examples/text_classification/model.pt --source-vocab serve/examples/text_classification/source_vocab.pt --handler text_classifier_kf --extra-files "serve/examples/text_classification/index_to_name.json"

	'''
 * Step - 3 : Start the Torchserve model using the below command :
	'''
	torchserve --start  --model-store model_store --ncs --models my_tc=my_text_classifier.mar
	
	'''
 * Step - 4 : Start the Postman Application and make a POST request to hit the inference endpoint

The Endpoint URL : http://127.0.0.1:8080/v1/models/my_tc:predict

The sample Request on the http client body is as below:

{
"instances":[{
            "name":"context",
            "data":"Bloomberg has reported on the economy",
            "target":0

  }]    
}

The Response is as below:


The response is as below :

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





	


