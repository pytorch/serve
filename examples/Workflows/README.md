# Workflow examples

Workflows can be used to compose an ensemble of Pytorch models and python functions. A workflow is executed as a DAG where the nodes can be either models packaged as mar files or function nodes specified in the workflow handler file. The DAG is used to define sequential or parallel pipelines.

Typically, the function nodes are used for processing or augmenting intermediate data or aggregating data from multiple nodes. Preprocessing nodes are used when we want to apply some common transformation to an input payload which is going to be passed to multiple model nodes. Aggregating data from multiple nodes allows you to build model ensembles which aggregate or vote on the results of multiple models.

<!-- It is also possible to use the same mar file in multiple workflows and register them at the same time. The model server will create separate instances of this model for the different workflows. This is demonstrated in the NMT Transformers example where the English-to-German model is used in both back translation and dual translation workflows. -->

We've included 2 reference examples including a sequential pipeline and parallel pipeline.

For a more detailed explanation of Workflows and what is currently supported please refer to the main [documentation](../../docs/workflows.md)

 * [Parallel workflow using nmt transformers example](nmt_transformers_pipeline/)
 * [Sequential workflow using resnet for dog breed classification](dog_breed_classification/)
