# Text Classification using TorchServe's default text_classifier handler

## !!!Deprecation Warning!!!
This example requires TorchText which is deprecated. Please use version <= 0.11.1 of TorchServe for this example

This is an example to create a text classification dataset and train a sentiment model. We have used the following torchtext example to train the model.

https://github.com/pytorch/text/tree/master/examples/text_classification

We have copied the files from above example and made small changes to save the model's state dict and added default values.

# Training the model

Run the following commands to train the model :

```bash
python run_script.py
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
    torchserve --start --model-store model_store --models my_tc=my_text_classifier.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/my_tc -T examples/text_classification/sample_text.txt
    ```
To make a captum explanations request on the Torchserve side, use the below command:

```bash
curl -X POST http://127.0.0.1:8080/explanations/my_tc -T examples/text_classification/sample_text.txt
```

In order to run Captum Explanations with the request input in a json file, follow the below steps:

In the config.properties, specify `service_envelope=body` and make the curl request as below:
```bash
curl -H "Content-Type: application/json" --data @examples/text_classification/text_classifier_ts.json http://127.0.0.1:8080/explanations/my_tc_explain
```
When a json file is passed as a request format to the curl, Torchserve unwraps the json file from the request body. This is the reason for specifying service_envelope=body in the config.properties file

### Captum Explanations

The explain is called with the following request api `http://127.0.0.1:8080/explanations/my_tc_explain`

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


NOTE:
The current default model for text classification uses EmbeddingBag which Computes sums or means of ‘bags’ of embeddings, without instantiating the intermediate embedding, so it returns the captum explanations on a sentence embedding level and not on a word embedding level.

### Running KServe

Refer the [End to End KServe document](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/README.md) to run it in the cluster.
