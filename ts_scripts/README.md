The scripts listed here execute on all of the supported platforms(Linux, Windows, OSX)
The scripts are designed to execute independent of each other.


1. [install_dependencies.py](install_dependencies.py) : Helps you install build and test dependencies for torchserve (like java, torch packages, newman, etc)
   ```
   # Example
   python ts_scripts/install_dependencies.py

2. [tsutils.py](tsutils.py.py) : Import this module in your scripts to have access to utility methods like start\stop torchserve, register\infer\unregister models, etc.
3. [shell_utils.py](shell_utils.py) : Import this module in your scripts to have access to utility methods like download file, remove directory, unzip files, etc
4. [install_from_src.py](install_from_src.py) : Installs torch and torch-model-archiver from source.
5. [test_frontend.py](test_frontend.py) : Executes gradle tests
6. [test_torchserve.py](test_torchserve.py) : Executes liniting in `ts` directory and executes pytests
7. [test_modelarchiver.py](test_modelarchiver.py) : Executes liniting in `model-archiver` directory and executes pytests (unit & integration tests)
8. [test_sanity.py](test_sanity.py) : Executes sanity tests (selected few models are registered\infered\unregistered) and markdown files are checked for broken links
9. [test_api.py](test_api.py) : Executes newman api collections (management\inference\increased_timeout_inference\https\all)
   ```
   # Example
   python ts_scripts/test_api.py management
   python ts_scripts/test_api.py inference
   python ts_scripts/test_api.py all
   ```
10. [test_regression.py](test_regression.py) : Executes regression pytests.