import transformers
from pathlib import Path
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
""" This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
"""
print('Transformers version',transformers.__version__)

def transformers_model_dowloader(pretrained_model_name = 'bert-base-uncased'):
    print("Download model and tokenizer", pretrained_model_name)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
    # A Fine_tunining process based on your needs can be added.
    # An example of Colab notebook for Fine_tunining process has been provided in the README.

    NEW_DIR = "./Transformer_model"
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

    print("Save model and tokenizer", pretrained_model_name, 'in directory', NEW_DIR)
    transformer_model.save_pretrained(NEW_DIR)
    transformer_tokenizer.save_pretrained(NEW_DIR)
if __name__== "__main__":
    transformers_model_dowloader(pretrained_model_name='bert-base-uncased')
