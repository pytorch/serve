# TorchServe on Windows

## Contents of this Document

* [Prerequisites](#prerequisites)
* [Install from binaries](#install-from-binaries)
* [Install from source](#install-from-source)

## Prerequisites

 - At present, it has only been certified on windows server 2019 however should work fine on Windows 10.
 - Make sure you are an admin user or have admin rights
 - The instruction given here will use anaconda Powershell terminal to install torchserve
 - Install Anaconda as given [here](https://docs.anaconda.com/anaconda/install/windows/)
 - Install Git as given [here](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe)
 - Install openjdk17
    - Download [openjdk17](https://download.oracle.com/java/17/archive/jdk-17.0.3_windows-x64_bin.zip)
    - Unzip and edit/add environment variables i.e. PATH and JAVA_HOME
    e.g.
    - Using command line `unzip jdk-17.0.3_windows-x64_bin.zip` or using GUI interface
    - Edit system or user profile environment variable `PATH` value and append path `<your-openjdk17-path>\bin` to it
 - Install nodejs
    - Download [nodejs](https://nodejs.org/dist/v14.15.1/node-v14.15.1-x64.msi)
    - Post installation make sure nodejs and npm node modules binaries are present in PATH environment variable.
    - You may have to re-start windows if your 'Anaconda Powershell Prompt' (APP) is not able to detect npm or nodejs commands

## Install from binaries

NOTE At present, wheels for windows are not available on PyPi. However following steps can also be used if you have prebuilt torchserve wheel for windows.

 - Start 'Anaconda Powershell Prompt' (APP) as Admin User i.e. By right click on APP and run following commands
 - `git clone https://github.com/pytorch/serve.git`
 - `cd serve`
 - `python .\ts_scripts\install_dependencies.py`
 - For local wheel file
    - `pip install <your-torchserve-wheel-file-name>.whl`
 - For PyPi package (N/A at present)
    - `pip install torchserve torch-model-archiver`
 - Start torchserve `torchserve.exe --start --model-store <path-to-model-store>`
 - For next steps refer [Serving a model](../docs/getting_started.md#serve-a-model)

## Install from source

 - Ensure that system or user profile environment variable name `JAVA_HOME` with value as `<your-openjdk17-path>` path is present.
 - Install [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-in/help/2977003/the-latest-supported-visual-c-downloads)

   NOTE ensure that you have restarted system after install above Visual C++ components
 - Ensure that 'nvidia-smi.exe' is available in `Path` environment variable. Usually, it should be available under `<your_install_drive>\Program Files\NVIDIA Corporation\NVSMI`
   e.g. C:\Program Files\NVIDIA Corporation\NVSMI, add this path to `Path` env variable
 - Start 'Anaconda Powershell Prompt' (APP) as Admin User i.e. By right click on APP and run following commands
 - `git clone https://github.com/pytorch/serve.git`
 - `pip install click`
 - `cd serve`

    #### For production usage, use commands below:
    - `python .\ts_scripts\install_dependencies.py --environment=prod`
    - `python .\ts_scripts\install_from_src.py`

    #### For development purposes, use commands below:
    If you plan to develop with TorchServe and change some source code, commands below will help.
    The install_dependencies script installs few extra dependencies which are needed for development and testing.
    - `python .\ts_scripts\install_dependencies.py --environment=dev`
    - `python .\ts_scripts\install_from_src.py`

## Troubleshooting
 - If you are building from source then you may have to change the port number for inference, management and metrics apis as specified in `frontend/server/src/test/resources/config.properties`,
   all files in `frontend/server/src/test/resources/snapshot/*` and `frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java`
 - If `curl` command fails to execute then run following command on APP (anaconda powershell prompt)
 `Remove-item alias:curl`
 Refer to this [SO answer](https://stackoverflow.com/questions/25044010/running-curl-on-64-bit-windows) for details.
