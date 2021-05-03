# Workflow examples

The following links provide examples on how to implement workflows with different models. The workflow feature can be used for serving an ensemble of models and python functions through workflow APIs. A workflow is executed as a DAG where the nodes can be either models (MAR files) or functions specified in the workflow handler file. The DAG need not contain additional functions if not required. Typically, the function nodes are used for processing or augmenting intermediate data or aggregating data from multiple nodes. Preprocessing nodes are used when we want to apply some common transformation to an input payload which is going to be passed to multiple model nodes. An example use case for this would be a preprocessing node passing transformed data to two branches with model nodes (refer dog/breed classification example below). In other cases, having a preprocessing step in the model handler itself might suffice.
Currently, only sequential workflow is supported fully.
For a more detailed explanation of Workflows and what is currently supported please refer to the main [documentation](../../docs/workflows.md)

 * [Pipeline/Sequential workflow using densenet161 image classifier example](densenet_image_classifier_pipeline/)
 * [Pipeline/Sequential workflow using nmt tranformers example](nmt_tranformers_pipeline/)
 * [Pipeline/Sequential workflow using resnet for dog breed classification](dog_breed_classification/)
