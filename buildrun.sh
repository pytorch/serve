torch-model-archiver --model-name iris --version 1.0 --model-file examples/iris/iris.py --serialized-file examples/iris/iris.pt --export-path model_store  --handler examples/iris/iris_handler.py --force
torchserve --start --ncs --foreground --model-store model_store --models iris.mar
