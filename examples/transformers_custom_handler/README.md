# Text Classfication with huggingface/transformers on the CoLA task using a custom model handler

This is an example to finetune and serve a BERT model from huggingface/transformers 
for the CoLA task (GLUE). 

The code for finetuning is based on the transformers GLUE CoLA example:
https://github.com/huggingface/transformers/tree/v2.8.0/examples#glue

Code for a simple custom handler for the resulting BERT model can be found in the 
file `transformers_cola_handler.py`.


# Finetuning

Run the following commands to finetune BERT:

```bash
./run_script.sh
```

The above command generated tokenizer and model checkpoints in directory `"./outputs"`.

# Serve the finetuned BERT classification model on TorchServe

 * Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name bert --version 1.0 --serialized-file outputs/pytorch_model.bin --handler "./transformers_cola_handler.py" --extra-files "./outputs/config.json,./outputs/vocab.txt"
    ```

 * Register the model on TorchServe using the above model archive file and run inference on a sample text
   
    ```bash
    mkdir model_store
    mv bert.mar model_store/
    torchserve --start --model-store model_store --models bert=bert.mar
    curl -X POST http://127.0.0.1:8080/predictions/bert -T sample_data/positive_cola_example
    ```
