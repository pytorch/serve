# Workflow pipeline example using densenet161 image classification model

The following examples splits the [densenet161 model serving example](../../image_classifier/densenet_161) into a pipeline workflow which has following three parts

 - Pre-processing (Workflow function): Converts the input image and converts it into tensor. Defined in [workflow handler](densenet_workflow_handler.py)
 - Densenet model (Workflow model): Runs inference on the pre-processed image. Defined in [workflow specification](densenet_workflow.yaml)
 - Post-processing (Workflow function): Maps the predicted classes to labels. Defined in [workflow handler](densenet_workflow_handler.py)

## Flow

 -> Input image -> Preprocessing -> Model inference -> Postprocessing -> Output

Refer `dag` section on [workflow specification](densenet_workflow.yaml)

## Commands to create a densenet161 model archive for workflow

Run the commands given in following steps from the current directory.

```bash
cd densenet_model
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet_wf --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --handler densenet_handler.py
mv densenet_wf.mar <path_to_model_store>
```

## Commands to create a workflow archive

Run the commands given in following steps from the current directory.

```bash
torch-workflow-archiver --workflow-name densenet --spec-file densenet_workflow.yaml --handler densenet_workflow_handler.py --extra-files index_to_name.json
mv densenet.war <path_to_workflow_store>
```

## Serve the workflow

Run the commands given in following steps from the current directory.

```
torchserve --start --model-store <path_to_model_store> --workflow-store <path_to_workflow_store> --ncs
curl -X POST "http://127.0.0.1:8081/workflows?url=densenet.war"
curl http://127.0.0.1:8080/wfpredict/densenet161 -T kitten.jpg
```