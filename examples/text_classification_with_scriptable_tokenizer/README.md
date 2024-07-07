# Text Classification using a Scriptable Tokenizer

## Deprecation Warning!
This example requires TorchText which is deprecated. Please use version <= 0.11.0 of TorchServe for this example

TorchScript is a way to serialize and optimize your PyTorch models.
A scriptable tokenizer is a special tokenizer which is compatible with [TorchScript's compiler](https://pytorch.org/docs/stable/jit.html) so that it can be jointly serialized with a PyTorch model.
When deploying an NLP model it is important to use the same tokenizer during training and inference to achieve the same model accuracy in both phases of the model live cycle.
Using a different tokenizer for inference than during training can decrease the model performance significantly.
Thus, is can be beneficial to combine the tokenizer together with the model into a single deployment artifact as it reduces the amount of preprocessing code in the handler leading to less synchronization effort between training and inference code bases.
This example shows how to combine a text classification model with a scriptable tokenizer into a single artifact and deploy it with TorchServe.
For demonstration purposes we use a pretrained model as created in this tutorial:

https://github.com/pytorch/text/blob/main/examples/tutorials/sst2_classification_non_distributed.py


# Training the Model

To train the model we need to follow the steps described in this [this tutorial](https://github.com/pytorch/text/blob/main/examples/tutorials/sst2_classification_non_distributed.py) and export the model weight into a ```model.pt``` file.
To use the SST-2 dataset torchtext requires torch.data to be installed.
This can be achieved with pip by running:

```
pip install torchdata
```

Or conda by running:

```
conda install -c pytorch torchdata
```

Subsequently, we need to add the command ```torch.save(model.state_dict(), "model.pt")``` at the end of the training script and then run it with:

```bash
python sst2_classification_non_distributed.py
```

A pretrained ```model.pt``` is also available for download [here](https://bert-mar-file.s3.us-west-2.amazonaws.com/text_classification_with_scriptable_tokenizer/model.pt).
The trained model can then be combined and compiled with TorchScript using the script_tokenizer_and_model.py script. Here ```model.pt``` are the model weights saved after training and ```model_jit.pt``` is the combination of tokenizer and model compiled with TorchScript.

```bash
python script_tokenizer_and_model.py model.pt model_jit.pt
```


# Serve the Text Classification Model on TorchServe

 * Create a torch model archive using the torch-model-archiver utility to archive the file created above.

    ```bash
    torch-model-archiver --model-name scriptable_tokenizer --version 1.0 --serialized-file model_jit.pt --handler handler.py --extra-files "index_to_name.json"
    ```

 * Register the model on TorchServe using the above model archive file and run a classification

    ```bash
    mkdir model_store
    mv scriptable_tokenizer.mar model_store/
    torchserve --start --model-store model_store --models my_tc=scriptable_tokenizer.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/my_tc -T sample_text.txt
    ```
 * Expected Output:
    ```
   {
       "Negative": 0.0972590446472168,
       "Positive": 0.9027408957481384
   }
   ```
