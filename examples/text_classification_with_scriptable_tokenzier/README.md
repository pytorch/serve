# Text Classfication using a Scriptable Tokenizer 

This example shows how to combine a text classification model with a scriptable tokenizer into a single artifact and deploy it with TorchServe.
The combination of model and tokenizer into a single artifact reduces the handler complexity and ensures that the same tokenizer is used during training and inference 
The training of the model was taken from this tutorial:

https://github.com/pytorch/text/blob/main/examples/tutorials/sst2_classification_non_distributed.py

which was adopted for this purpose.

# Training the Model

To use the SST-2 dataset torchtext requires torch.data to be installed.
This can be achieved with pip by running:

```
pip install torchdata
```

Or conda by runnning:

```
conda install -c pytorch torchdata
```

Subsequently we can run the training script with this command:

```bash
python train_model.py
```

The above command trains the model and combines it with a the scriptable tokenizer.
The combination is then scripted with TorchScript and saved into a model.pt file.

# Serve the Text Classification Model on TorchServe

 * Create a torch model archive using the torch-model-archiver utility to archive the file created above.
 
    ```bash
    torch-model-archiver --model-name scriptable_tokenizer --version 1.0 --serialized-file model.pt --handler handler.py --extra-files "index_to_name.json"
    ```
       
 * Register the model on TorchServe using the above model archive file and run a classification
   
    ```bash
    mkdir model_store
    mv scriptable_tokenizer.mar model_store/
    torchserve --start --model-store model_store --models my_tc=scriptable_tokenizer.mar
    curl http://127.0.0.1:8080/predictions/my_tc -T sample_text.txt
    ```
