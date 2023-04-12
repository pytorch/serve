import os

import torch
from Transformer_handler_generalized import TransformersSeqClassifierHandler

if "NEURON_RT_NUM_CORES" not in os.environ:
    os.environ["NEURON_RT_NUM_CORES"] = "1"


class TransformersSeqClassifierNeuronHandler(TransformersSeqClassifierHandler):
    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        input_ids_batch, attention_mask_batch = input_batch
        num_inferences = len(input_ids_batch)
        batch_size = int(self.setup_config.get("batch_size", "1"))

        # insert padding if a partial batch was received
        padding = batch_size - num_inferences
        if padding > 0:
            pad = torch.nn.ConstantPad1d((0, 0, 0, padding), value=0)
            input_ids_batch = pad(input_ids_batch)
            attention_mask_batch = pad(attention_mask_batch)

        inferences = super().inference((input_ids_batch, attention_mask_batch))

        return inferences[:num_inferences]
