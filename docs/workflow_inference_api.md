# Workflow Inference API

Workflow Inference API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

The TorchServe server supports the following APIs:

* [Predictions API](#predictions-api) - Gets predictions from the served model

## Predictions API

To get predictions from a workflow, make a REST call to `/wfpredict/{workflow_name}`:

`POST /wfpredict/{workflow_name}`

### curl Example

```bash
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

curl http://localhost:8080/wfpredict/myworkflow -T kitten_small.jpg
```

The result is JSON object returning the response bytes from the leaf node of the workflow DAG.
