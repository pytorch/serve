# Running Stable diffusion model using Huggingface diffusers in Torchserve.

### Step 1: Download model

Set access token generated form Huggingface in `Download_model.py` file

```bash
python Download_model.py
```

### Step 2: Compress downloaded model

```bash
cd Diffusion_model
zip -r ../model.zip *
```

### Step 3: Generate MAR file

Navigate back to `diffusers` directory.

```bash
torch-model-archiver --model-name stable-diffusion --version 1.0 --handler stable_diffusion_handler.py --extra-files model.zip -r requirements.txt
```

### Step 4: Start torchserve

```bash
torchserve --start --ts-config config.properties
```

### Step 5: Run inference

```bash
curl -v http://localhost:8080/predictions/stable-diffusion -T sample.txt > output.txt
```

Note: `sample_v1.json` and `sample_v2.json` are kserve inputs.

### Step 6: Restore image

```python
import json
from PIL import Image
import numpy as np

# read file
with open('example.json', 'r') as myfile:
    data=myfile.read()

# parse file
json_data = json.loads(data)
new_image = Image.fromarray(np.array(json.loads(json_data["predictions"]), dtype='uint8'))
```
