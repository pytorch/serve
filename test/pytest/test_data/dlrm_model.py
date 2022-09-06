from dlrm_factory import DLRMFactory
from dlrm_model_config import DLRMModelConfig
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES


def simple_dlrm_model_config():
    """
    This simplified configuration is used during unittesting.
    """

    return DLRMModelConfig(
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


class Model(DLRMFactory):
    """
    To save time during the unit test we exchange the standard DLRM model from this example with a smaller version.
    """

    def __new__(cls):
        model_config = simple_dlrm_model_config()
        return super().__new__(cls, model_config)
