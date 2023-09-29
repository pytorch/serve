# Code Coverage

## To check branch stability run the sanity suite as follows
 - Install dependencies (if not already installed)
   For CPU:
   ```bash
   python ts_scripts/install_dependencies.py --environment=dev
   ```

   For GPU
    - Install dependencies (if not already installed)
   ```bash
   python ts_scripts/install_dependencies.py --environment=dev --cuda=cu102
   ```
   > Supported cuda versions as cu118, cu117, cu116, cu113, cu111, cu102, cu101, cu92

 - Execute sanity suite
   ```bash
   python ./torchserve_sanity.py
   ```

## To run frontend build suite run following command

```bash
frontend/gradlew -p frontend clean build
```

TorchServe frontend build suite consists of :

  * checkstyle
  * findbugs
  * PMD
  * UT

The reports are generated at following path : `frontend/server/build/reports`

## To run backend pytest suite run following command

```bash
python -m pytest --cov-report html:htmlcov --cov=ts/ ts/tests/unit_tests/
```

The reports are generated at following path : `htmlcov/`

## To run python linting on `ts` package run following command

```bash
pylint -rn --rcfile=./ts/tests/pylintrc ts/.
```

## To run pytest suite on model-archiver run following command

```bash
cd model-archiver
python -m pytest --cov-report html:htmlcov_ut --cov=model_archiver/ model_archiver/tests/unit_tests/
```

The reports are generated at following path : `model-archiver/htmlcov_ut/`

## To run IT suite on model-archiver run following command

```bash
cd model-archiver
pip install .
python -m pytest --cov-report html:htmlcov_it --cov=model_archiver/ model_archiver/tests/integ_tests/
```

The reports are generated at following path : `model-archiver/htmlcov_it/`

**Note**: All the above commands needs to be excuted from serve home

## To run the markdown link check run following command

### Install markdown link checker dependencies
Run following commands to install node and [`markdown-link-check`](https://github.com/tcort/markdown-link-check/) npm package.

* Install node

For Linux :
```
sudo apt-get -y install nodejs-dev
sudo apt-get -y install npm
```

For Mac :
```
brew install node
brew install npm
```

* Install markdown

```
sudo npm install -g n
sudo npm install -g markdown-link-check
```
Execute this command to run markdown link checks locally. It will check broken links in all files with ".md" extension in current directory recursively.
```
for i in **/*.md; do markdown-link-check $i --config link_check_config.json; done;
```
