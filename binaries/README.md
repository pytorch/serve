# Building TorchServe and Torch-Model-Archiver release binaries 
1. Make sure all the dependencies are installed
   ```bash
   python ts_scripts/install_dependencies.py --environment dev
   ```
   > For GPU with Cuda 10.1, make sure add the `--cuda cu101` arg to the above command
   
2. To build a `torchserve` and `torch-model-archiver` wheel execute:
   ```bash
   python build.py
   ```
   > If the scripts detect a conda environment, it also builds torchserve conda packages  
   > For additional info on conda builds refer to [this readme](conda/README.md)
3. Build outputs are located at
   - Wheel files
     `dist/torchserve-*.whl`  
     `model-archiver/dist/torch_model_archiver-*.whl`  
   - Conda pacakages
     `binaries/conda/output/*`  

# Install torchserve and torch-model-archiver binaries
1. To install torchserve using the newly created binaries execute:
   ```bash
   python install.py
   ```
2. Alternatively, you can manually install binaries
   - Using wheel files
      ```bash
      pip install dist/torchserve-*.whl
      pip install model-archiver/dist/torch_model_archiver-*.whl
      ```
   - Using conda packages
     ```bash
      conda install --channel binaries/conda/output -y torchserve torch-model-archiver
     ```