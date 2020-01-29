import torch.nn as nn

r"""
The model is composed of the embeddingbag layer and the linear layer.

nn.EmbeddingBag computes the mean of 'bags' of embeddings. The text
entries here have different lengths. nn.EmbeddingBag requires no
padding because the lengths of sentences are saved in offsets.
Therefore, this method is much faster than the original one
with TorchText Iterator and Batch.

Additionally, since it accumulates the average across the embeddings on the fly,
nn.EmbeddingBag can enhance the performance and memory efficiency
to process a sequence of tensors.

"""


class TextSentiment(nn.Module):
    def __init__(self, vocab_size=1308844, embed_dim=32, num_class=4):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        r"""
        Arguments:
            text: 1-D tensor representing a bag of text tensors
            offsets: a list of offsets to delimit the 1-D text tensor
                into the individual sequences.
        """
        return self.fc(self.embedding(text, offsets))
