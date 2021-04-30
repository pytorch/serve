## Serving Huggingface Transformers using TorchServe

In this example, we show how to serve a Fine_tuned or off-the-shelf Transformer model from [huggingface](https://huggingface.co/transformers/index.html) using TorchServe. We use a custom handler, Transformer_handler.py. This handler enables us to use pre-trained transformer models from Hugginface, such as BERT, RoBERTA, XLM, etc. for use-cases defined in the AutoModel class such as AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification.

We borrowed ideas to write a custom handler for transformers from tutorial presented in [mnist from image classifiers examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist) and the post by [MFreidank](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18).

First, we need to make sure that have installed the Transformers, it can be installed using the following  command.

 `pip install transformers`

### Objectives
1. Demonstrate how to package a transformer model with custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

### Serving a Model using TorchServe

To serve a model using TrochServe following steps are required.

-  Frist, preparing the requirements for packaging a model, including serialized model, and other required files. Torchserve supports, models in eager mode as well as Torchscript.
- Create a torch model archive using the "torch-model-archiver" utility to archive the above files along with a handler ( in this example custom handler for transformers) .
- Register the model on TorchServe using the above model archive file and run the inference.

### **Getting Started with the Demo**

There are two paths to obtain the required model files for this demo.

- **Option A** : To yield desired results, one should fine-tuned each of the intended models to use before hand and saving the model and tokenizer using "save_pretrained() ". This will result in pytorch_model.bin file along with vocab.txt and config.json files. These files should be moved to a folder named "Transformer_model" in the current directory. Alternatively, one can opt to use a Torchscript for the serialized model. This is explained in the next section.
- **Option B**: There is another option just for demonstration purposes, to simply run "Download_Transformer_models.py", . The  "Download_Transformer_models.py" script loads and saves the required files mentioned above in  "Transformer_model" directory, using a setup config file, "setup_config.json". Also, settings in  "setup_config.json", are used in the handler, "Transformer_handler_generalized.py", as well to operate on the selected mode and other related settings.

#### Torchscript Support

[Torchscript](https://pytorch.org/docs/stable/jit.html#creating-torchscript-code) along with Pytorch JIT are designed to provide portability and perfromance for Pytorch models. Torchscript is a static subset of Python language that capture the structure of Pytroch programs and JIT uses this structure for optimization.

Torchscript exposes two APIs, script and trace, using any of these APIs, on the regular Pytorch model developed in python, compiles it to Torchscript. The resulted Torchscript can be loaded in a process where there is no Python dependency. The important difference between trace and script APIs, is that trace does not capture parts of the model which has data dependency such as control flow, this is where script is a better choice.

To create Torchscript from Huggingface Transformers, torch.jit.trace() will be used that returns an executable or [`ScriptFunction`](https://pytorch.org/docs/stable/jit.html#torch.jit.ScriptFunction) that will be optimized using just-in-time compilation. We need to provide example inputs, torch.jit.trace, will record the operations performed on all the tensors when running the inputs through the transformer models. This option can be chosen through the setup_config.json by setting *save_mode* : "torchscript". We need to keep this in mind, as torch.jit.trace()  record operations on tensors,  the size of inputs should be the same both in tracing and when using it for inference, otherwise it will raise an error. Also, there is torchscript flag that needs to be set when setting the configs to load the pretrained models, you can read more about it in this [Huggingface's doc](https://huggingface.co/transformers/torchscript.html).

Here is how Huggingface transfomers can be converted to Torchscript using the trace API, this has been shown in download_Transformer_models.py as well:

First of all when setting the configs, the torchscript flag should be set :

`config = AutoConfig.from_pretrained(pretrained_model_name,torchscript=True)`

When the model is loaded, we need a dummy input to pass it through the model and record the operations using the trace API:

```
dummy_input = "This is a dummy input for torch jit trace"
inputs = tokenizer.encode_plus(dummy_input,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
input_ids = inputs["input_ids"]
traced_model = torch.jit.trace(model, [input_ids])
torch.jit.save(traced_model,os.path.join(NEW_DIR, "traced_model.pt"))

```

#### Setting the setup_config.json

In the setup_config.json :

*model_name* : bert-base-uncased , roberta-base or other available pre-trained models.

*mode:* "sequence_classification "for sequence classification, "question_answering"for question answering and "token_classification" for token classification.

*do_lower_case* : True or False for use of the Tokenizer.

*num_labels* : number of outputs for "sequence_classification", or "token_classification".

*save_mode* : "torchscript" or "pretrained", this setting will be used by "Download_transformer_models.py" script as well as the handler, to download/save and load the model in Torchscript or save_pretrained mode respectively.

*max_length* : maximum length for the  input sequences to the models, this will be used in preprocessing of the hanlder. Also, if you choose to use Torchscript as the serialized model  for packaging your model this length should be equal to the length that has been used during the tracing of the model using torch.jit.trace.

Once, setup_config.json has been set properly, the next step is to run " Download_Transformer_models.py":

`python Download_Transformer_models.py`

This produces all the required files for packaging using a huggingface transformer model off-the-shelf without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory. 



#### Setting the extra_files

There are few files that are used for model packaging and at the inference time. "index_to_name.json"  is passed as extra file to the model archiver and used for mapping predictions to labels. "sample_text.txt", is used at the inference time to pass the text that we want to get the inference on. In case vocab related files are not passed extra files, the handler will load the tokenizer from a pre_trained model. 

In case of having a customized vocab or having vocab.json that need the tokenizer work with, it is required to pass all other tokenizer related files such tokenizer_config.json, special_tokens_map.json, config.json and if available merges.txt.

index_to_name.json for question answering is not required.

If intended to use Transformer handler for Token classification, the index_to_name.json should be formatted as follows for example:

`{"label_list":"[O, B-MISC, I-MISC, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC]"}`

To use Transformer handler for question answering, the sample_text.txt should be formatted as follows.:

`{"question" :"Who was Jim Henson?", "context": "Jim Henson was a nice puppet"}`

"question" represents the question to be asked from the source text named as "context" here. It is recommended that based on the task in hand, modify the preprocessing step in the handler to read the context from a pickled file or any other format that is applicable. Follwoing this change  then you can simply  change the format of this query as well.

### Creating a torch Model Archive

Once, setup_config.json,  sample_text.txt and index_to_name.json are set properly, we can go ahead and package the model and start serving it. The artifacts realted to each operation mode (such as sample_text.txt, index_to_name.json) can be place in their respective folder. The current setting in "setup_config.json" is based on "bert-base-uncased" off the shelf, for sequence classification and Torchscript as save_mode. To fine-tune BERT, RoBERTa or other models, for question ansewering you can refer to [squad example](https://huggingface.co/transformers/examples.html#squad) from huggingface. Model packaging using pretrained for save_mode can be done as follows:

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

In case of using Torchscript the packaging step would look like the following:

```
torch-model-archiver --model-name BERTSeqClassification_Torchscript --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

As a reminder, if you are using this handler for sequence or token classification, it's needed to pass the label mapping of the predictions, "index_to_name.json", through the  --extra-files as well.

An example of passing customized vocab file for the tokenizer is:

```
torch-model-archiver --model-name BERTSeqClassification_Torchscript --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./vocab.json,./config.json,./special_tokens_map.json,./tokenizer_config.json,./merges.txt,./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

As a reminder, if model checkpoints are saved in pytorch_model.bin ("saved_mode" should be set to "pretrained" in the setup_config.json) you need to pass it instead of traced_model.pt. 

### Registering the Model on TorchServe and Running Inference

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv BERTSeqClassification_Torchscript.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification_Torchscript.mar

```

- To run the inference using our registered model, open a new terminal and run: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T ./Seq_classification_artifacts/sample_text.txt`



For captum Explanations on the Torchserve side, use the below curl request:
```bash
curl -X POST http://127.0.0.1:8080/explanations/my_tc -T ./Seq_classification_artifacts/sample_text.txt
```

In order to run Captum Explanations with the request input in a json file, follow the below steps:

In the config.properties, specify `service_envelope=body` and make the curl request as below:
```bash
curl -H "Content-Type: application/json" --data @examples/Huggingface_Transformers/bert_ts.json http://127.0.0.1:8080/explanations/bert_explain
```

When a json file is passed as a request format to the curl, Torchserve unwraps the json file from the request body. This is the reason for specifying service_envelope=body in the config.properties file

### Registering the Model on TorchServe and Running batch Inference

The following uses .mar file created from  model packaging using pretrained for save_mode to register the model for batch inference on sequence classification, by setting the batch_size when registering the model.

```
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store 

curl -X POST "localhost:8081/models?model_name=BERT_seq_Classification&url=BERTSeqClassification.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```

Now to run the batch inference follwoing command can be used:

`curl -X POST http://127.0.0.1:8080/predictions/BERT_seq_Classification  -T ./Seq_classification_artifacts/sample_text1.txt& curl -X POST http://127.0.0.1:8080/predictions/BERT_seq_Classification  -T ./Seq_classification_artifacts/sample_text2.txt& curl -X POST http://127.0.0.1:8080/predictions/BERT_seq_Classification -T ./Seq_classification_artifacts/sample_text3.txt&`

--- 
**NOTE** Batches is not currently implemented for explanations 
---

### Captum Explanations

The explain is called with the following request api http://127.0.0.1:8080/explanations/bert_explain

Torchserve supports Captum Explanations for Eager models only.

Captum/Explain doesn't support batching.

#### The handler changes:

1. The handlers should initialize.
```python
self.lig = LayerIntegratedGradients(captum_sequence_forward, self.model.bert.embeddings) 
```
in the initialize function for the captum to work.

2. The Base handler handle uses the explain_handle method to perform captum insights based on whether user wants predictions or explanations. These methods can be overriden to make your changes in the handler.

3. The get_insights method in the handler is called by the explain_handle method to calculate insights using captum.

4. If the custom handler overrides handle function of base handler, the explain_handle function should be called to get captum insights.

Functions for captum like construct_input_ref, captum_sequence_forward, summarize_attributions, get_word_token should be implemented.

### Implementation Approach for captum batches
Batches is not currently implemented for explanations 
The batch inputs, target and preprocessed inputs can be stored in a list in the explain handle function and directly passed to the get_insights function of the handler.

As captum makes many predictions for each sample, there may be a timeout for the response when batching is implemented. 

### Captum Explanations for Visual Insights

The [Captum Explanations for Visual Insights Notebook](../../captum/Captum_visualization_for_bert.ipynb) gives an insight into how the captum explanations can be used to visually represent the attributions and word importances. The pre-requisite is to have the prediction response ready. In this example, the prediction response from the BERT Seq Classification is used. 

### Running KFServing 
Refer the [BERT Readme for KFServing](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/Huggingface_readme.md) to run it locally.

Refer the [End to End KFServing document](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/README.md) to run it in the cluster.