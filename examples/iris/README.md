# Example to use torchserve as inference server for TensorBoard's What-If-Tool plugin

1. Clone the repository. Assume that the cloned folder location is `/Users/username/git/serve` 
2. Compile the model: `mkdir model_store && torch-model-archiver --model-name iris --version 1.0 --model-file examples/iris/iris.py --serialized-file examples/iris/iris.pt --export-path model_store  --handler examples/iris/iris_handler.py --force`. A new file `/Users/username/git/serve/model_store/iris.mar` should be generated.
3. Launch the model server: `docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 --mount type=bind,source=/Users/username/git/serve/model_store,target=/tmp/models pytorch/torchserve:latest torchserve --model-store=/tmp/models --models iris.mar`

4. `pip install tensorboard-plugin-wit==1.7.0 tensorflow==2.4.1 pycurl` (Tested with Python 3.8.8)

5. Start TensorBoard server: `tensorboard --logdir . --whatif-use-unsafe-custom-prediction examples/iris/custom_wit_predict_fn.py`

6. Navigate to `http://localhost:6006/#whatif&p.whatif.examplesPath=%2FUsers%2Fusername%2Fgit%2Fserve%2Fexamples%2Firis%2Firis.csv&p.whatif.modelName1=iris&p.whatif.inferenceAddress1=localhost%3A8080` to see the result.



