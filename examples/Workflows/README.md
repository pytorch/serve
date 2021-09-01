# Workflow examples

Workflows can be used to compose an ensemble of Pytorch models and Python functions and package them in a `war` file. A workflow is executed as a DAG where the nodes can be either Pytorch models packaged as `mar` files or function nodes specified in the workflow handler file. The DAG can be used to define both sequential or parallel pipelines.

As an example a sequential pipeline may look something like

```
input -> function1 -> model1 -> model2 -> function2 -> output
```

And a parallel pipeline may look something like 

```
                          model1
                         /       \
input -> preprocessing ->         -> aggregate_func
                         \       /
                          model2
```

You can experiment with much more complicated workflows by configuring a `YAML` file. We've included 2 reference examples including a sequential pipeline and parallel pipeline.
* [Parallel workflow using nmt transformers example](nmt_transformers_pipeline/)
* [Sequential workflow using resnet for dog breed classification](dog_breed_classification/)

For a more detailed explanation of Workflows and what is currently supported please refer to the main [documentation](../../docs/workflows.md)
