from ts.handler_utils.utils import import_class
from ts.torch_handler.distributed.base_neuronx_continuous_batching_handler import (
    BaseNeuronXContinuousBatchingHandler,
)


class LlamaContinuousBatchingHandler(BaseNeuronXContinuousBatchingHandler):
    def __init__(self):
        super(BaseNeuronXContinuousBatchingHandler, self).__init__()
        self.model_class = import_class(
            class_name="llama.model.LlamaForSampling",
            module_prefix="transformers_neuronx",
        )

        self.tokenizer_class = import_class(
            class_name="transformers.LlamaTokenizer",
        )
