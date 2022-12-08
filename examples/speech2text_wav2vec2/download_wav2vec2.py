#!/usr/bin/env python3

from transformers import AutoModelForCTC, AutoProcessor
import os

modelname = "facebook/wav2vec2-base-960h"
model = AutoModelForCTC.from_pretrained(modelname)
processor = AutoProcessor.from_pretrained(modelname)

modelpath = "model"
os.makedirs(modelpath, exist_ok=True)
model.save_pretrained(modelpath)
processor.save_pretrained(modelpath)
