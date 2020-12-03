The scripts listed here execute on all of the supported platforms(Linux, Windows, OSX)


1. [install_dependencies.py](install_dependencies.py) : Helps you install build and test dependencies for torchserve (like java, torch packages, newman, etc)*[]: 
   You can use argument `--cuda=cu101` to select cuda 10.1. This is pre-requisite script for executing following scripts -
   - serve/torchserve_sanity.py
   - serve/ts_script/install_from_src.py
   - serve/test/regression_tests.py
   
   ```
   # Example: Use following command to install cuda 10.1 related dependencies,
   python ts_scripts/install_dependencies.py --cuda=cu101

2. [install_from_src.py](install_from_src.py) : Installs torch and torch-model-archiver from source. 
   NOTE Before executing this script, you will need to install dependencies using `ts_scripts/install_dependencies.py` script.
