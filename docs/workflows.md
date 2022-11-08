# TorchServe Workflows

TorchServe can be used for serving ensemble of Pytorch models packaged as mar files and Python functions through Workflow APIs. 

It utilizes [REST based APIs](rest_api.md) for workflow management and predictions.

A Workflow is served on TorchServe using a workflow-archive(.war) which comprises of following: 

## Workflow Specification file

A workflow specification is a YAML file which provides the details of the models to be executed and a DAG for defining data flow.

The YAML file is split into a few sections
1. `models` which include global model parameters
2. `m1,m2,m3` all the relevant model parameters which would override the global model params
3. `dag` which describes the structure of the workflow, which nodes feed into which other nodes

E.g.

```yaml
models:
    #global model params
    min-workers: 1
    max-workers: 4
    batch-size: 3
    max-batch-delay : 5000
    retry-attempts : 3
    timeout-ms : 5000
    m1:
       url : model1.mar #local or public URI
       min-workers: 1   #override the global params
       max-workers: 2
       batch-size: 4
     
    m2:
       url : model2.mar

    m3:
       url : model3.mar
       batch-size: 3

    m4:
      url : model4.mar
 
dag:
  pre_processing : [m1]
  m1 : [m2]
  m2 : [m3]
  m3 : [m4]
  m4 : [postprocessing]
```

### Workflow Models

The `models` section of the workflow specification defines the models used in the workflow. It uses the following syntax:

```
models:
    <model_name>:
        url: <local or public url for mar file>
```

### Workflow model properties

User can define following workflow model properties:

| Properties | Description | Default value |
| --- | --- | --- |
| min-workers | Number of minimum workers launched for every workflow model | 1 |
| max-workers | Number of maximum workers launched for every workflow model | 1 |
| batch-size | Batch size used for every workflow model | 1 |
| max-batch-delay | Maximum batch delay time TorchServe waits for every workflow model to receive `batch_size` number of requests.| 50 ms |
| retry-attempts | Retry attempts for a specific workflow node in case of a failure | 1 |
| timeout-ms | Timeout in MilliSeconds for a given node | 10000 |

These properties can be defined as a global value for every model and can be over-ridden at every model level in workflow specification. Refer the above example for more details.

## Workflow DAG

User can define the dataflow of a workflow using the `dag` section of the workflow specification. The `dag` consists of the model names defined in the `model` section and python function names which are implemented in the workflow-archive's handler file.

### Sequential DAG

Eg.
```
dag:
  function1 : [model1]
  model1 : [model2]
  model2 : [function2]
```

Which maps to this data flow

```
input -> function1 -> model1 -> model2 -> function2 -> output
```

### Parallel DAG

E.g
```
dag:
  pre_processing: [model1, model2]
  model1: [aggregate_func]
  model2: [aggregate_func]
```

Which maps to this data flow

```
                          model1
                         /       \
input -> preprocessing ->         -> aggregate_func
                         \       /
                          model2
```

## Handler file

A handler file (python) is supplied in the workflow archive (.war) and consists of all the functions used in the workflow dag.

Eg.
```python
def preprocess(data, context):
    pass

def postprocess(data, context):
    pass

```

## Related docs
* [workflow_inference_api.md](workflow_inference_api.md)
* [workflow_management_api.md](workflow_management_api.md)

## Known issues

* Each workflow dag node (model/function) will receive input as bytes
* Only following output types are supported by workflow models/functions : String, Int, List, Dict of String, int, Json serializable objects, byte array and Torch Tensors
* Workflow scale/updates is not supported through APIs. User will need to unregister the workflow and re-register with the required changes
* Snapshots are not supported for workflows and related models are not captured in the workflow
* Workflow versioning is not supported
* Workflows registration having public model URL with mar file names which are already registered will fail.