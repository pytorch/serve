# Testing TorchServe

## Pre-requisites

You will need some additional Python modules to run the unit tests and linting.

```bash
pip install mock pytest pylint pytest-mock pytest-cov
```

You will also need the source for the project, so clone the project first.

```bash
git clone https://github.com/pytorch/serve.git
cd serve
```

## Unit Tests

You can run the unit tests with the following:

```bash
python -m pytest ts/tests/unit_tests/
```

To get the coverage report of unit tests, you can run :

```bash
python -m pytest --cov-report term-missing --cov=ts/ ts/tests/unit_tests/
```

or:

```bash
python -m pytest --cov-report html:htmlcov --cov=ts/ ts/tests/unit_tests/
```

## Lint test

You can run the lint tests with the following:

```bash
pylint -rn --rcfile=./ts/tests/pylintrc ts/.
```
