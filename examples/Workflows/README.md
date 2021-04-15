# Workflow examples

The following links provide examples on how to implement workflows with different models. The workflow feature can be used for serving an ensemble of models and python functions through workflow APIs. A workflow is executed as a DAG where the nodes can be either model (MAR files) or functions specified in the workflow handler file. The DAG need not contain additional functions if not required. Typically, the function nodes are used for processing or augmenting intermediate data or aggregating data from multiple nodes. Currently, only sequential workflow is supported fully.
For a more detailed explanation of Workflows and what is currently supported please refer to the main [documentation](../../docs/workflows.md)

 * [Pipeline/Sequential workflow using densenet161 image classifier example](densenet_image_classifier_pipeline/)
 * [Pipeline/Sequential workflow using nmt tranformers example](nmt_tranformers_pipeline/)
 * [Pipeline/Sequential workflow using resnet for dog breed classification](dog_breed_classification/)
