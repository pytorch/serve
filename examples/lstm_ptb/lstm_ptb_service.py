import json
import os

import mxnet as mx

from mxnet_utils import nlp
from model_handler import ModelHandler


class MXNetLSTMService(ModelHandler):
    """
    MXNetLSTMService service class. This service consumes a sentence
    from length 0 to 60 and generates a sentence with the same size.
    """

    def __init__(self):
        super(MXNetLSTMService, self).__init__()
        self.mxnet_ctx = None
        self.mx_model = None
        self.labels = None
        self.signature = None
        self.data_names = None
        self.data_shapes = None
        self.epoch = 100

        self.buckets = [10, 20, 30, 40, 50, 60]
        self.start_label = 1
        self.invalid_key = "\n"
        self.invalid_label = 0
        self.layout = "NT"
        self.vocab = {}
        self.idx2word = {}

    def initialize(self, context):
        super(MXNetLSTMService, self).initialize(context)

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        batch_size = properties.get("batch_size")
        if batch_size > 1:
            raise ValueError("Batch is not supported.")

        # reading signature.json file
        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")

        with open(signature_file_path) as f:
            self.signature = json.load(f)

        self.data_names = []
        self.data_shapes = []
        for input_data in self.signature["inputs"]:
            self.data_names.append(input_data["data_name"])
            self.data_shapes.append((input_data['data_name'], tuple(input_data['data_shape'])))

        # reading vocab_dict.txt file
        vocab_dict_file = os.path.join(model_dir, "vocab_dict.txt")
        with open(vocab_dict_file, 'r') as vocab_file:
            self.vocab[self.invalid_key] = self.invalid_label
            for line in vocab_file:
                word_index = line.split(' ')
                if len(word_index) < 2 or word_index[0] == '':
                    continue
                self.vocab[word_index[0]] = int(word_index[1].rstrip())
        for key, val in self.vocab.items():
            self.idx2word[val] = key

        # Load pre-trained lstm bucketing module
        num_layers = 2
        num_hidden = 200
        num_embed = 200

        stack = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, mode="lstm").unfuse()

        # Define symbol generation function for bucket module
        def sym_gen(seq_len):
            data = mx.sym.Variable("data")
            embed = mx.sym.Embedding(data=data, input_dim=len(self.vocab),
                                     output_dim=num_embed, name="embed")

            stack.reset()
            outputs, _ = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

            pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
            pred = mx.sym.FullyConnected(data=pred, num_hidden=len(self.vocab), name="pred")
            pred = mx.sym.softmax(pred, name='softmax')

            return pred, ('data',), None

        self.mxnet_ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        # Create bucketing module and load weights
        self.mx_model = mx.mod.BucketingModule(
            sym_gen=sym_gen,
            default_bucket_key=max(self.buckets),
            context=self.mxnet_ctx)

        checkpoint_prefix = "{}/{}".format(model_dir, "lstm_ptb")

        self.mx_model.bind(data_shapes=self.data_shapes, for_training=False)

        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, checkpoint_prefix, self.epoch)
        self.mx_model.set_params(arg_params, aux_params)

    def preprocess(self, data):
        """
        This service doesn't support batch, always get data from first item.

        :param data:
        :return:
        """
        input_data = data[0].get("data")
        if input_data is None:
            input_data = data[0].get("body")

        # Convert a string of sentence to a list of string
        sent = input_data[0]["input_sentence"].lower().split(" ")
        assert len(sent) <= self.buckets[-1], "Sentence length must be no greater than %d." % (self.buckets[-1])
        # Encode sentence to a list of int
        res, _ = nlp.encode_sentences(
            [sent], vocab=self.vocab, start_label=self.start_label,
            invalid_label=self.invalid_label)

        return res

    def inference(self, data):
        data_batch = nlp.pad_sentence(
            data[0], self.buckets, invalid_label=self.invalid_label,
            data_name=self.data_names[0], layout=self.layout)
        self.mx_model.forward(data_batch)
        return self.mx_model.get_outputs()

    def postprocess(self, data):
        # Generate predicted sentences
        word_idx = mx.nd.argmax(data[0], axis=1).asnumpy()
        res = ""
        for idx in word_idx:
            res += self.idx2word[idx] + " "

        ret = {"prediction": res}
        return [ret]


# Following code is not necessary if your service class contains `handle(self, data, context)` function
_service = MXNetLSTMService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
