# Testing MMS

## Pre-requisites

You will need some additional Python modules to run the unit tests and linting.

```bash
pip install mock pytest pylint
```

You will also need the source for the project, so clone the project first.

```bash
git clone https://github.com/awslabs/mxnet-model-server.git
cd mxnet-model-server
```

## Unit Tests

You can run the unit tests with the following:

```bash
python -m pytest mms/tests/unit_tests/
```

To get the coverage report of unit tests, you can run :

```bash
python -m pytest --cov-report term-missing --cov=mms/ mms/tests/unit_tests/
```

or:

```bash
python -m pytest --cov-report html:htmlcov --cov=mms/ mms/tests/unit_tests/
```

## Lint test

You can run the lint tests with the following:

```bash
pylint -rn --rcfile=./mms/tests/pylintrc mms/.
```
