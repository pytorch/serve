import os
import subprocess

os.makedirs(".data", exist_ok=True)

cmd = "python train.py AG_NEWS --device cpu --save-model-path  model.pt --dictionary source_vocab.pt"
subprocess.run(cmd, shell=True, check=True)
