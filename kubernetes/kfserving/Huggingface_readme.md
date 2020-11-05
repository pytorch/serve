# Serve a BERT Model for Inference and Explanations on the KFServing side :

In this document, the .mar file creation, request & response on the KFServing side and the KFServing changes to the handler files for BERT Sequence Classification model using a custom handler.


## .mar file creation

The .mar file creation command is as below:

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler Transformer_model/Transformer_handler_generalized.py --source-vocab Transformer_model/vocab.txt --extra-files "Transformer_model/config.json,Transformer_model/setup_config.json,Transformer_model/index_to_name.json"
```


## Request and Response

The curl request for inference is as below:
```
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:predict
```

The Prediction response is as below :

```
{
	"predictions" : [
		
		"Accepted"
	]
}
```
The curl request for explanations is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:explain
```

The Explanation response is as below :

```bash
{
  "explanations": [
    {
      "importances": [
        0.0,
        -0.6324464886855062,
        -0.03311582455592391,
        0.26816950103292936,
        -0.29124773714666485,
        0.5422589134206756,
        -0.3848765077942276,
        0.0
      ],
      "words": [
        "[CLS]",
        "bloomberg",
        "has",
        "reported",
        "on",
        "the",
        "economy",
        "[SEP]"
      ]
    }
  ]
}
```


## KFServing changes to the handler files



* 1) When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.

    The bert request difference between the regular torchserve and kfserving is as below

    ### Regular torchserve request:
	```
	[
		{
			"data" : "The recent climate change across world is impacting negatively"
		}     
	]
	```

    ### KFServing Request:
	```
	{

		"instances":[
						{
							"data" : "The recent climate change across world is impacting negatively"
						}
					]
	}
	```

    The KFServing request is unwrapped by the kfserving envelope in torchserve  and sent like a torchserve request. So effectively the values of  `instances`  key is sent to the handlers.

        

* 2)The Request data for kfserving  is a batches of dicts as opposed to batches of bytes array(text file) in the 	 regular torchserve.

    So in the preprocess method of [Transformer_handler_generalized.py](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py), KFServing doesn't require the data to be utf-8 decoded for text inputs, hence the code was modified to ensure that Torchserve Input Requests which are sent as text file are only utf-8 decoded and not for the KFServing Input Requests.








