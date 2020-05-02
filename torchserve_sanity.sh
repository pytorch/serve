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
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://torchserve.s3.amazonaws.com/mar_files/$1.mar&initial_workers=4&synchronous=true")

  if [ ! "$response" == 200 ]
  then
      echo "Failed to register $1 model with torchserve"
      cleanup
      exit 1
  else
      echo "Successfully registered $1 model with torchserve"
  fi
}

unregister_model()
{
  echo "Registering resnet-18 model"
  response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X DELETE "http://localhost:8081/models/$1")

  if [ ! "$response" == 200 ]
  then
      echo "Failed to register $1 model with torchserve"
      cleanup
      exit 1
  else
      echo "Successfully registered $1 model with torchserve"
  fi
}

run_inference()
{
  for i in {1..4}
  do
    echo "Running inference on $1 model"
    response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST http://localhost:8080/predictions/$1 -T $2)

    if [ ! "$response" == 200 ]
    then
        echo "Failed to run inference on $1 model"
        cleanup
        exit 1
    else
        echo "Successfully ran infernece on $1 model."
    fi
  done

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

# run object detection example

register_model "fastrcnn"

run_inference "fastrcnn" "examples/object_detector/persons.jpg"

unregister_model "fastrcnn"

# run image segmentation example

register_model "fcn_resnet_101"

run_inference "fcn_resnet_101" "examples/image_segmenter/fcn/persons.jpg"

unregister_model "fcn_resnet_101"

# run text classification example -

register_model "my_text_classifier"

run_inference "my_text_classifier" "examples/text_classification/sample_text.txt"

unregister_model "my_text_classifier"

# run image classification example

register_model "resnet-18"

run_inference "resnet-18" "examples/image_classifier/kitten.jpg"

stop_torchserve

# restarting torchserve
# this should restart with the generated snapshot and resnet-18 model should be automatically registered

start_torchserve

run_inference

stop_torchserve

cleanup

echo "CONGRATULATIONS!!! YOUR BRANCH IS IN STABLE STATE"
