import transformers
from pathlib import Path
import os
import json
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,
 AutoModelForTokenClassification, AutoConfig)
from transformers import set_seed

NEW_DIR = 'DirWhereYouWantToSaveModelForMAR'
 
print('Transformers version',transformers.__version__)
set_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("elgeish/cs224n-squad2.0-albert-base-v2")

model = AutoModelForQuestionAnswering.from_pretrained("elgeish/cs224n-squad2.0-albert-base-v2")

model.save_pretrained(NEW_DIR)
tokenizer.save_pretrained(NEW_DIR)
