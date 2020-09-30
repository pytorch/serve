# Building TorchServe and Torch-Model-Archiver release binaries 

1. To produce a `torchserve` and `torch-model-archiver` wheel execute:
   ```bash
   python build.py
   ```
   > If the scripts detect a conda environment, it also builds torchserve conda packages  
   > For additional info on conda builds refer to [this readme](conda/README.md)
2. The wheel files are located at
   - `dist/torchserve-*.whl`
   - `model-archiver/dist/torch_model_archiver-*.whl`
3. The conda pacakages are located at
   - `binaries/conda/output/*`

# Install torchserve and torch-model-archiver binaries
1. To install torchserve using the newly created binaries execute:
   ```bash
   python install.py
   ```
2. Alternatively, you can manuaaly install binaries
   - Using wheel files
      ```bash
      pip install dist/torchserve-*.whl
      pip install model-archiver/dist/torch_model_archiver-*.whl
      ```
   - Using conda packages
     ```bash
      conda install --channel binaries/conda/output -y torchserve torch-model-archiver
     ```