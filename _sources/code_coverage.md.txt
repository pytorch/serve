## To check branch stability run the sanity suite as follows

 - For CPU or GPU with Cuda 10.2
 
```bash
./torchserve_sanity.sh
```
 - For GPU with Cuda 10.1
 ```bash
./torchserve_sanity.sh cuda101
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
