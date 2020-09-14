# TorchServe on Windows

## Contents of this Document

* [Prerequisites](#prerequisites)
* [Manual installation from source](#manual_installation_from_source)
* [Install from binaries](#install-from-binaries)
* [Install from source](#install-from-source)

## Prerequisites

 - Make sure you are an admin user or have admin rights
 - The instruction given here will use anaconda Powershell terminal to install torchserve
 - Install Anaconda as given [here](https://docs.anaconda.com/anaconda/install/windows/)
 - Install Git as given [here](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe)
 - Install openjdk11
    - Download [openjdk11](https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_windows-x64_bin.zip)
    - Unzip and edit/add environment variables i.e. PATH and JAVA_HOME
    e.g.
    - Using command line `unzip openjdk-11*_bin.zip` or using GUI interface
    - Edit system or user profile environment variable `PATH` value and append path `<your-openjdk11-path>\bin`

NOTE torchserve has been tested on Windows10 64bit system

## Manual installation from source -  This for development and inital hands-on eventually it will be removed in the final commit for this feature

 - Add new system or user profile environment variable name `JAVA_HOME` and value as `<your-openjdk11-path>`
 - Install [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-in/help/2977003/the-latest-supported-visual-c-downloads)
 
   NOTE ensure that you have restarted system after install above Visual C++ components
 - Start 'Anaconda Powershell Prompt' (APP) as Admin User i.e. By right click on APP and run following commands
 - `git clone https://github.com/pytorch/serve.git`
 - Install torchserve 
     - `cd serve`
     - `pip install -U -r requirements/developer.txt`
     - `pip install -U -r requirements/cpu.txt -f  https://download.pytorch.org/whl/torch_stable.html``
     - `pip install .`
     - Start torchserve `torchserve.exe --start --model-store <path-to-model-store>`
     - For next steps refer [Serving a model](https://github.com/pytorch/serve#serve-a-model)
  - Install torch-model-archiver
    - `cd model-archiver`
    - `pip install .`
    - torch-model-archiver will be installed as exe hence add .exe extension to given examples in repo.
    - For next steps refer [mar commands](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-command-line-interface)
  - For doing build and UT
    - `pip install -U -r requirements/developer.txt`
    - `pip install -U -r requirements/cpu.txt -f  https://download.pytorch.org/whl/torch_stable.html`
    - `cd frontend`
    - `.\gradlew.bat clean build`

## Install from binaries [WIP]

NOTE At present, wheels for windows are not available on PyPi. However following steps can also be used if you have prebuilt torchserve wheel for windows.

 - Start 'Anaconda Powershell Prompt' (APP) as Admin User i.e. By right click on APP and run following commands
 - `git clone https://github.com/pytorch/serve.git`
 - `cd serve`
 - `pip install -U -r requirements.txt`
 - For local wheel file
    - `pip install <your-torchserve-wheel-file-name>.whl`
 - For PyPi package (N/A at present)
    - `pip install torchserve torch-model-archiver`
 - Start torchserve `torchserve.exe --start --model-store <path-to-model-store>`
 - For next steps refer [Serving a model](https://github.com/pytorch/serve#serve-a-model)
    
## Install from source [WIP]

 - Add new system or user profile environment variable name `JAVA_HOME` and value as `<your-openjdk11-path>`
 - Install [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/en-in/help/2977003/the-latest-supported-visual-c-downloads)
 
   NOTE ensure that you have restarted system after install above Visual C++ components
 - Start 'Anaconda Powershell Prompt' (APP) as Admin User i.e. By right click on APP and run following commands
 - `git clone https://github.com/pytorch/serve.git`
 - `cd serve`
 - `pip install -U -r requirements/developer.txt`
 - Refer [Install torchserve for development](https://github.com/pytorch/serve#install-torchserve-for-development)

## Troubleshooting
 - If you are building from source then you may have to change the port number for inference, management and metrics apis as specified in `frontend/server/src/test/resources/config.properties`,
   all files in `frontend/server/src/test/resources/snapshot/*` and `frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java`
 - If `curl` command fails to execute then run following command on APP (anaconda powershell promopt)
 `Remove-item alias:curl`
 Refer this [SO answer](https://stackoverflow.com/questions/25044010/running-curl-on-64-bit-windows) for details.
