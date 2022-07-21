from dataclasses import dataclass
from typing import List

from dlrm_factory import DLRMFactory
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES


class Model(DLRMFactory):
    def __new__(cls):
        @dataclass
        class DLRMModelConfig:
            dense_arch_layer_sizes: List[int]
            dense_in_features: int
            embedding_dim: int
            id_list_features_keys: List[str]
            num_embeddings_per_feature: List[int]
            over_arch_layer_sizes: List[int]

        model_config = DLRMModelConfig(
            dense_arch_layer_sizes=[32, 16, 8],
            dense_in_features=len(DEFAULT_INT_NAMES),
            embedding_dim=8,
            id_list_features_keys=DEFAULT_CAT_NAMES,
            num_embeddings_per_feature=len(DEFAULT_CAT_NAMES)
            * [
                3,
            ],
            over_arch_layer_sizes=[32, 32, 16, 1],
        )
        return super().__new__(cls, model_config)
