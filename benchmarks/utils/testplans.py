# Test plans (soak, vgg11_1000r_10c,  vgg11_10000r_100c,...)
def soak(execution_params):
    execution_params["requests"] = 100000
    execution_params["concurrency"] = 10


def vgg11_1000r_10c(execution_params):
    execution_params["url"] = "https://torchserve.pytorch.org/mar_files/vgg11.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10


def vgg11_10000r_100c(execution_params):
    execution_params["url"] = "https://torchserve.pytorch.org/mar_files/vgg11.mar"
    execution_params["requests"] = 10000
    execution_params["concurrency"] = 100


def resnet152_batch(execution_params):
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4


def resnet152_batch_docker(execution_params):
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/resnet-152-batch.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4
    execution_params["exec_env"] = "docker"


def bert_batch(execution_params):
    execution_params[
        "url"
    ] = "https://torchserve.pytorch.org/mar_files/BERTSeqClassification.mar"
    execution_params["requests"] = 1000
    execution_params["concurrency"] = 10
    execution_params["batch_size"] = 4
    execution_params[
        "input"
    ] = "../examples/Huggingface_Transformers/Seq_classification_artifacts/sample_text.txt"


def workflow_nmt(execution_params):
    pass


def custom(execution_params):
    pass


update_plan_params = {
    "soak": soak,
    "vgg11_1000r_10c": vgg11_1000r_10c,
    "vgg11_10000r_100c": vgg11_10000r_100c,
    "resnet152_batch": resnet152_batch,
    "resnet152_batch_docker": resnet152_batch_docker,
    "bert_batch": bert_batch,
    "workflow_nmt": workflow_nmt,
    "custom": custom,
}
