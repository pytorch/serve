# Building TorchServe and Torch-Model-Archiver release binaries 
1. Make sure all the dependencies are installed
   ##### Linux and macOS:
   ```bash
   python ts_scripts/install_dependencies.py --environment=dev
   ```

   ##### Windows:
   ```pwsh
   python .\ts_scripts\install_dependencies.py --environment=dev
   ```
   > For GPU with Cuda 10.2, make sure add the `--cuda cu102` arg to the above command
   
   
2. To build a `torchserve` and `torch-model-archiver` wheel execute:
   ##### Linux and macOS:
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
    ##### Linux and macOS:
   - Wheel files
     `dist/torchserve-*.whl`  
     `model-archiver/dist/torch_model_archiver-*.whl`
     `workflow-archiver/dist/torch_workflow_archiver-*.whl`
   - Conda pacakages
     `binaries/conda/output/*`  
     
    ##### Windows:
    - Wheel files
      `dist\torchserve-*.whl`  
      `model-archiver\dist\torch_model_archiver-*.whl`  
      `workflow-archiver\dist\torch_workflow_archiver-*.whl`  
    - Conda pacakages
      `binaries\conda\output\*`

# Install torchserve and torch-model-archiver binaries
1. To install torchserve using the newly created binaries execute:
    ##### Linux and macOS:
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
      ##### Linux and macOS:
      ```bash
      pip install dist/torchserve-*.whl
      pip install model-archiver/dist/torch_model_archiver-*.whl
      pip install workflow-archiver/dist/torch_workflow_archiver-*.whl
      ```

      ##### Windows:
      ```pwsh
      pip install .\dist\<torchserve_wheel>
      pip install .\model-archiver\dist\<torch_model_archiver_wheel>
      pip install .\workflow-archiver\dist\<torch_workflow_archiver_wheel>
      ```
   - Using conda packages
      ##### Linux and macOS:
     ```bash
      conda install --channel ./binaries/conda/output -y torchserve torch-model-archiver torch-workflow-archiver
     ```
    
     ##### Windows:
     Conda install is currently not supported. Please use pip install command instead.

# Building TorchServe, Torch-Model-Archiver & Torch-WorkFlow-Archiver nightly binaries
1. Make sure all the dependencies are installed
   ##### Linux and macOS:
   ```bash
   python ts_scripts/install_dependencies.py --environment=dev
   ```

   ##### Windows:
   ```pwsh
   python .\ts_scripts\install_dependencies.py --environment=dev
   ```
   > For GPU with Cuda 10.2, make sure add the `--cuda cu102` arg to the above command


2. To build a `torchserve`, `torch-model-archiver` & `torch-workflow-archiver` nightly wheel execute:
   ##### Linux and macOS:
   ```bash
   python binaries/build.py --nightly
   ```
   ##### Windows:
   ```pwsh
   python .\binaries\build.py --nightly
   ```

   > If the scripts detect a conda environment, it also builds torchserve conda packages
   > For additional info on conda builds refer to [this readme](conda/README.md)

3. Build outputs are located at
    ##### Linux and macOS:
   - Wheel files
     `dist/torchserve-*.whl`
     `model-archiver/dist/torch_model_archiver-*.whl`
     `workflow-archiver/dist/torch_workflow_archiver-*.whl`
   - Conda pacakages
     `binaries/conda/output/*`

    ##### Windows:
    - Wheel files
      `dist\torchserve-*.whl`
      `model-archiver\dist\torch_model_archiver-*.whl`
      `workflow-archiver\dist\torch_workflow_archiver-*.whl`
    - Conda pacakages
      `binaries\conda\output\*`

4. Nightly binary formats
   #### PyPI binaries
   -  The binary has the format `<binary_name>-<YYYY.M.D>-py<x>-none-any.whl`
   - For example: `torchserve_nightly-2022.7.12-py3-none-any.whl `

   #### Conda binaries
   - The binary has the format `<binary_name>-<version>.dev<YYYYMMDD>-py<xx>_0.tar.bz2`
   - For example: `torchserve-0.6.0.dev20220713-py39_0.tar.bz2`
   - This is similar to the format of other pytorch domain nightly binaries at https://anaconda.org/pytorch-nightly/repo

# Uploading packages for testing to a personal account
1. Export the following environment variables for TestPypi and anaconda.org authentication
   ```
   export CONDA_TOKEN=<>
   export TWINE_USERNAME=<>
   export TWINE_PASSWORD=<>
   ```
2. Edit `upload.py` to change the CONDA_USER if necessary
3. Run the following commands to build the packages, and then upload them to staging repos
   ```
   python3 ts_scripts/install_dependencies.py --environment=dev
   python3 binaries/conda/build_packages.py --install-conda-dependencies
   exec bash
   python3 binaries/build.py
   cd binaries/
   python3 upload.py --upload-pypi-packages --upload-conda-packages 
   ```
4. To upload *.whl files to S3 bucket, run the following command: 
   Note: `--nightly` option puts the *.whl files in a subfolder named 'nightly' in the specified bucket
   ```
   python s3_binary_upload.py --s3-bucket <s3_bucket> --s3-backup-bucket <s3_backup_bucket> --nightly
   ```

## Uploading packages to production torchserve account

As a first step binaries and docker containers need to be available in some staging environment. In that scenario the binaries can just be `wget`'d and then uploaded using the instructions below and the docker staging environment just needs a 1 line code change in https://github.com/pytorch/serve/blob/master/docker/promote-docker.sh#L8

### pypi
Binaries should show up here: https://pypi.org/project/torchserve/

You need to be on the list of maintainers to run the below.

`twine upload <path/to.wheel>`


### conda
Binaries should show up here: https://anaconda.org/pytorch/torchserve


```
# Authenticate with pytorchbot credentials
anaconda login

# Upload binaries
anaconda upload -u pytorch <path/to/.bz2>
```

## docker
Binaries should show up here: https://hub.docker.com/r/pytorch/torchserve

Change the staging org to your personal docker or test docker account https://github.com/pytorch/serve/blob/master/docker/promote-docker.sh#L8


### Direct upload

To build a docker image follow instructionss from [docker/README.md](../docker/README.md)

Once the image is built make sure you give it the correct name with
`docker tag <your_repository:your_image_name/> pytorch/torchserve:<tag>`

For an official release our tags include `pytorch/torchserve/<version_number>-cpu`, `pytorch/torchserve/<version_number>-gpu`, `pytorch/torchserve/latest-cpu`, `pytorch/torchserve/latest-gpu`

## Direct upload Kserve
To build the Kserve docker image follow instructions from [kubernetes/kserve](../kubernetes/kserve/README.md)

When tagging images for an official release make sure to tag with the following format `pytorch/torchserve-kfs/<version_number>-cpu` and `pytorch/torchserve-kfs/<version_number>-gpu`. 

### Uploading from staging account

```
# authenticate to docker with pytorchbot credentials
docker login

# upload docker image
./docker/promote-docker.sh
```

If everything looks good then run `$DRY_RUN=disabled ./docker/promote-docker.sh`
