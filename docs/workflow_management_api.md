# Management API

TorchServe provides the following APIs that allows you to manage workflows at runtime:

1. [Register a workflow](#register-a-workflow)
1. [Describe a workflow's status](#describe-workflow)
1. [Unregister a workflow](#unregister-a-workflow)
1. [List registered workflows](#list-workflows)

The Workflow Management API listens on port 8081 and is only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

## Register a workflow

`POST /workflows`

* `url` - Workflow archive download url. Supports the following locations:
  * a local workflow archive (.war); the file must be in the `workflow_store` folder (and not in a subfolder).
  * a URI using the HTTP(s) protocol. TorchServe can download `.war` files from the Internet.
* `workflow_name` - the name of the workflow; this name will be used as {workflow_name} in other APIs as part of the path. If this parameter is not present, `modelName` in MANIFEST.json will be used.

```bash
curl -X POST  "http://localhost:8081/workflows?url=https://<public_url>/myworkflow.mar"

{
  "status": "Workflow myworkflow has been registered and scaled successfully."
}
```

The workflow registration API parses the workflow specification file (.yaml) supplied in the workflow archive(.war) and registers all the models specified in the DAG with TorchServe using the provided configuration in the specification.

## Describe workflow

`GET /workflows/{workflow_name}`

Use the Describe Workflow API to get detail of a workflow:

```bash
curl http://localhost:8081/workflows/myworkflow
[
  {
    "workflowName": "myworkflow",
    "workflowUrl": "myworkflow.war",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 8,
    "maxBatchDelay": 5000,
    "workflowDag": "{preprocessing=[m1], m1=[postprocessing]}"
  }
]
```

## Unregister a workflow

`DELETE /workflows/{workflow_name}`

Use the Unregister Workflow API to free up system resources by unregistering a workflow from TorchServe:

```bash
curl -X DELETE http://localhost:8081/workflows/myworkflow

{
  "status": "Workflow \"myworkflow\" unregistered"
}
```

## List workflows

`GET /models`

* `limit` - (optional) the maximum number of items to return. It is passed as a query parameter. The default value is `100`.
* `next_page_token` - (optional) queries for next page. It is passed as a query parameter. This value is return by a previous API call.

Use the  list Workflows API to query currently registered workflows:

```bash
curl "http://localhost:8081/workflows"
```

This API supports pagination:

```bash
curl "http://localhost:8081/workflows?limit=2&next_page_token=2"

{
  "nextPageToken": "4",
  "workflows": [
    {
      "workflowName": "myworkflow1",
      "workflowUrl": "myworkflow1.war"
    },
    {
      "workflowName": "myworkflow2",
      "workflowUrl": "myworkflow2.war"
    }
  ]
}
```
