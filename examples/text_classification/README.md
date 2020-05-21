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
    torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file model.py --serialized-file model.pt --source-vocab source_vocab.pt --handler text_classifier --extra-files index_to_name.json
    ```
   
 * Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv my_text_classifier.mar model_store/
    torchserve --start --model-store model_store --models my_tc=my_text_classifier.mar
    curl http://127.0.0.1:8080/predictions/my_tc -T examples/text_classification/sample_text.txt
    ```
