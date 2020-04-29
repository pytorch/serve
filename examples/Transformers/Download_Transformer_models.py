import transformers
from pathlib import Path
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print('Transformers version',transformers.__version__) # Current version: 2.3.0

def transformers_model_dowloader(pretrained_model_name = 'bert-base-uncased'):
    print("Download model and tokenizer", pretrained_model_name)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
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
