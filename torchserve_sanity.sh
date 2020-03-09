cd frontend

if ./gradlew clean build;
then
  echo "Frontend build suite execution successfully"
else
  echo "Frontend build suite execution failed!!! Check logs for more details"
  exit 1
fi

cd ..
if python -m pytest --cov-report html:htmlcov --cov=ts/ ts/tests/unit_tests/;
then
  echo "Backend test suite execution successfully"
else
  echo "Backend test suite execution failed!!! Check logs for more details"
  exit 1
fi

cd model-archiver
if python -m pytest --cov-report html:htmlcov --cov=model_archiver/ model_archiver/tests/unit_tests/;
then
  echo "Model-archiver test suite execution successfully"
else
  echo "Model-archiver test suite execution failed!!! Check logs for more details"
  exit 1
fi