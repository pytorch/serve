# Building TorchServe and Torch-Model-Archiver release binaries 
1. Make sure all the dependencies are installed
   ##### Linux and MacOs:
   ```bash
   python ts_scripts/install_dependencies.py --environment=dev
   ```

   ##### Windows:
   ```pwsh
   python .\ts_scripts\install_dependencies.py --environment=dev
   ```
   > For GPU with Cuda 10.1, make sure add the `--cuda cu101` arg to the above command
   
   
2. To build a `torchserve` and `torch-model-archiver` wheel execute:
   ##### Linux and MacOs:
   ```bash
   python binaries/build.py
   ```
   ##### Windows:
   ```pwsh
   python .\binaries\build.py
   ```

   > If the scripts detect a conda environment, it also builds torchserve conda packages  
   > For additional info on conda builds refer to [this readme](conda/README.md)

3. Build outputs are located at
    ##### Linux and MacOs:
   - Wheel files
     `dist/torchserve-*.whl`  
     `model-archiver/dist/torch_model_archiver-*.whl`  
   - Conda pacakages
     `binaries/conda/output/*`  
     
    ##### Windows:
    - Wheel files
      `dist\torchserve-*.whl`  
      `model-archiver\dist\torch_model_archiver-*.whl`  
    - Conda pacakages
      `binaries\conda\output\*`

# Install torchserve and torch-model-archiver binaries
1. To install torchserve using the newly created binaries execute:
    ##### Linux and MacOs:
   ```bash
   python binaries/install.py
   ```

   ##### Windows:
   Note: If you have conda installed please do a manual install using Pip as shown below in option 2.
   ```pwsh
   python .\binaries\install.py
   ```
2. Alternatively, you can manually install binaries
   - Using wheel files
      ##### Linux and MacOs:
      ```bash
      pip install dist/torchserve-*.whl
      pip install model-archiver/dist/torch_model_archiver-*.whl
      ```

      ##### Windows:
      ```pwsh
      pip install .\dist\<torchserve_wheel>
      pip install .\model-archiver\dist\<torch_model_archiver_wheel>
      ```
   - Using conda packages
      ##### Linux and MacOs:
     ```bash
      conda install --channel ./binaries/conda/output -y torchserve torch-model-archiver
     ```
    
     ##### Windows:
     Conda install is currently not supported. Please use pip install command instead.
