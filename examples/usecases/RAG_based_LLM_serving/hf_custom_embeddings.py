from typing import Any, List

import torch
import torch._inductor.config as config
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer

# Enable AWS Graviton specific torch.compile optimizations
config.cpp.weight_prepack = True
config.freezing = True


class CustomEmbedding(HuggingFaceEmbeddings):
    tokenizer: Any

    def __init__(self, model_path: str, compile=True):
        """Initialize the sentence_transformer."""
        super().__init__()

        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.client = AutoModel.from_pretrained(model_path)

        if compile:
            self.client = torch.compile(self.client)

    class Config:
        arbitrary_types_allowed = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        embeddings = self.client(**encoded_input)
        embeddings = embeddings.pooler_output.detach().numpy()

        return embeddings.tolist()
