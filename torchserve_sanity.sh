#!/bin/bash
set -euxo pipefail

start_torchserve()
{
  echo "Starting TorchServe"
  torchserve --start --model-store model_store &
  pid=$!
  count=$(ps -A| grep $pid |wc -l)
  if [[ $count -eq 1 ]]
  then
          if wait $pid; then
                  echo "Successfully started TorchServe"
          else
                  echo "TorchServe start failed (returned $?)"
                  exit 1
          fi
  else
          echo "Successfully started TorchServe"
  fi

  sleep 10
}

stop_torchserve()
{
  torchserve --stop
  sleep 10
}

register_model()
{
  echo "Registering resnet-18 model"
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar&initial_workers=1&synchronous=true")

  if [ ! "$response" == 200 ]
  then
      echo "Failed to register model with torchserve"
      cleanup
      exit 1
  else
      echo "Successfully registered resnet-18 model with torchserve"
  fi
}

run_inference()
{
  echo "Running inference on resnet-18 model"
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST http://localhost:8080/predictions/resnet-18 -T examples/image_classifier/kitten.jpg)

  if [ ! "$response" == 200 ]
  then
      echo "Failed to run inference on resnet-18 model"
      cleanup
      exit 1
  else
      echo "Successfully ran infernece on resnet-18 model."
  fi
}

cleanup()
{
  stop_torchserve

  rm -rf model_store

  rm -rf logs
}


pip install mock pytest pylint pytest-mock pytest-cov

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

pip uninstall --yes torchserve
pip uninstall --yes torch-model-archiver

if pip install .;
then
  echo "Successfully installed TorchServe"
else
  echo "TorchServe installation failed"
  exit 1
fi

cd model-archiver

if python -m pytest --cov-report html:htmlcov --cov=model_archiver/ model_archiver/tests/unit_tests/;
then
  echo "Model-archiver UT test suite execution successfully"
else
  echo "Model-archiver UT test suite execution failed!!! Check logs for more details"
  exit 1
fi

if pip install .;
then
  echo "Successfully installed torch-model-archiver"
else
  echo "torch-model-archiver installation failed"
  exit 1
fi

if python -m pytest --cov-report html:htmlcov --cov=model_archiver/ model_archiver/tests/integ_tests/;
then
  echo "Model-archiver IT test suite execution successful"
else
  echo "Model-archiver IT test suite execution failed!!! Check logs for more details"
  exit 1
fi

cd ..

mkdir -p model_store

start_torchserve

register_model

stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference

cleanup

echo "CONGRATULATIONS!!! YOUR BRANCH IS IN STABLE STATE"
