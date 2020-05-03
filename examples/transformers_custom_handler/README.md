# Text Classfication with huggingface/transformers on the CoLA task using a custom model handler

This is an example to finetune and serve a BERT model from huggingface/transformers 
for the CoLA task (GLUE). 

The code for finetuning is based on the transformers GLUE CoLA example:
https://github.com/huggingface/transformers/tree/v2.8.0/examples#glue

Code for a simple custom handler for the resulting BERT model can be found in the 
file `transformers_cola_handler.py`.

## Installing dependencies

```bash
pip3 install transformers==2.8.0 sklearn==0.22.2 numpy==1.18.1
```

## Downloading the pretrained checkpoint

First, we download a pretrained checkpoint for a DistilBERT model 
finetuned for the SST sentiment analysis task:

```bash
python3 download_pretrained_model.py
```

The above command generates tokenizer and model checkpoints in directory `"./pretrained_model_checkpoint"`.

## Serve the sentiment analysis model using TorchServe

 * Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
```bash
torch-model-archiver --model-name distilbert --version 1.0 --serialized-file pretrained_model_checkpoint/pytorch_model.bin --handler "./transformers_custom_handler.py" --extra-files "./pretrained_model_checkpoint/config.json,./pretrained_model_checkpoint/vocab.txt,index_to_name.json"
```

* Register the sentiment classifier model on TorchServe using the above model archive file
   
```bash
mkdir model_store
mv distilbert.mar model_store/
torchserve --start --model-store model_store --models distilbert=distilbert.mar
```

* Run inference on sample text

We can call the Inference API with sample text to get sentiment predictions from our model:

```bash
curl -X POST http://127.0.0.1:8080/predictions/distilbert -T sample_data/positive_sentiment_example
```

This will print `"Positive Sentiment"`.

We can also try a negative example:
```bash
curl -X POST http://127.0.0.1:8080/predictions/distilbert -T sample_data/positive_sentiment_example
```
prints `"Negative Sentiment"`.
