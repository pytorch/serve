
## Commands to create the models and the workflow
The notebooks for training the [dog-cat classification](cat_dog_classification.ipynb) model and the [dog breed classification](dog_breed_classification.ipynb) models are provided in thos example. Once the models are trained you can download the corresponding .pth files and use them to generate the mar files for serving inference requests as below.

```
$ cd $TORCH_SERVE_DIR/examples/dog_breed_classification
$ mkdir model_store wf_store
$ torch-model-archiver -f --model-name cat_dog_classification --version 1.0 --model-file cat_dog_classification_arch.py --serialized-file /<path_to_model_files>/cat_dog_classification.pth --handler cat_dog_classification_handler.py --export-path model_store
$ torch-model-archiver -f --model-name dog_breed_classification --version 1.0 --model-file dog_breed_classification_arch.py --serialized-file /<path_to_model_files>/dog_breed_classification.pth --handler dog_breed_classification_handler.py --extra-files index_to_name.json --export-path model_store
$ torch-workflow-archiver -f --workflow-name dog_breed_wf --spec-file workflow_dog_breed_classification.yaml --handler workflow_dog_breed_classification_handler.py --export-path wf_store/
```

## Serve the workflow
```
$ torchserve --start --model-store model_store/ --workflow-store wf_store/ --ncs
$ curl -X POST "http://127.0.0.1:8081/workflows?url=dog_breed_wf.war"
{
  "status": "Workflow dog_breed_wf has been registered and scaled successfully."
}

$ curl https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg -o Cat.jpg
$ curl http://127.0.0.1:8080/wfpredict/dog_breed_wf -T Cat.jpg
It's a cat!

$ curl https://raw.githubusercontent.com/udacity/dog-project/master/images/Labrador_retriever_06457.jpg -o Dog.jpg
$ curl http://127.0.0.1:8080/wfpredict/dog_breed_wf -T Dog.jpg
{
  "Kuvasz": 0.9941568374633789,
  "American_water_spaniel": 0.0041659059934318066,
  "Glen_of_imaal_terrier": 0.0014263634802773595,
  "Cavalier_king_charles_spaniel": 0.0001453325676266104,
  "Plott": 2.3177999537438154e-05
}

$ curl http://127.0.0.1:8080/wfpredict/dog_breed_wf -T Dog.jpg -o 1.txt& \
curl http://127.0.0.1:8080/wfpredict/dog_breed_wf -T Cat.jpg -o 2.txt& \
curl http://127.0.0.1:8080/wfpredict/dog_breed_wf -T Dog.jpg -o 3.txt

$ cat *.txt
{
  "Akita": 0.7205144762992859,
  "Flat-coated_retriever": 0.09710536897182465,
  "Dachshund": 0.07697659730911255,
  "Cane_corso": 0.032701969146728516,
  "Chesapeake_bay_retriever": 0.01897248439490795
}It's a cat!{
  "Akita": 0.7205144762992859,
  "Flat-coated_retriever": 0.09710536897182465,
  "Dachshund": 0.07697659730911255,
  "Cane_corso": 0.032701969146728516,
  "Chesapeake_bay_retriever": 0.01897248439490795
}

$ curl -X DELETE "http://127.0.0.1:8081/workflows/dog_breed_wf"
{
  "status": "Workflow \"dog_breed_wf\" unregistered"
}
```
