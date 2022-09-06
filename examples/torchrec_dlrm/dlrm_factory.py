"""
This fectory creates a DLRM model with the Torchrec library compatible with the Criteo dataset format.
For this example we use an untrained model(model with random weights). More information on how to train this model can be found at
https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/
"""


import torch
import torchrec.distributed as trec_dist
from dlrm_model_config import DLRMModelConfig
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.types import ShardingType
from torchrec.inference.modules import quantize_embeddings
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


def create_default_model_config():

    return DLRMModelConfig(
        dense_arch_layer_sizes=[512, 256, 64],
        dense_in_features=len(DEFAULT_INT_NAMES),
        embedding_dim=64,
        id_list_features_keys=DEFAULT_CAT_NAMES,
        num_embeddings_per_feature=[
            45833188,
            36746,
            17245,
            7413,
            20243,
            3,
            7114,
            1441,
            62,
            29275261,
            1572176,
            345138,
            10,
            2209,
            11267,
            128,
            4,
            974,
            14,
            48937457,
            11316796,
            40094537,
            452104,
            12606,
            104,
            35,
        ],
        over_arch_layer_sizes=[512, 512, 256, 1],
    )


class DLRMFactory(type):
    def __new__(cls, model_config=None):

        # We use only a single GPU for this example
        world_size = 1

        # If we do not provide a model config we use the default one compatible with the Criteo dataset
        if not model_config:
            model_config = create_default_model_config()

        default_cuda_rank = 0
        device = torch.device("cuda", default_cuda_rank)
        torch.cuda.set_device(device)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=model_config.embedding_dim,
                num_embeddings=model_config.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                model_config.id_list_features_keys
            )
        ]
        # Creates an EmbeddingBagCollection without allocating any memory
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=model_config.dense_in_features,
            dense_arch_layer_sizes=model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=model_config.over_arch_layer_sizes,
            dense_device=device,
        )

        sharders = [
            QuantEmbeddingBagCollectionSharder(),
        ]

        constraints = {}
        for feature_name in model_config.id_list_features_keys:
            constraints[f"t_{feature_name}"] = ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.QUANT.value],
            )

        module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)

        # The planner will decide how the model memory will be allocated.
        # In case of multiple GPU (not part of this example) it decides how to split the model between the GPUs
        plan = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                compute_device="cuda",
                local_world_size=world_size,
            ),
            constraints=constraints,
        ).plan(module, sharders)

        # This step brings it all together and finally allocates the memory for the model
        module = trec_dist.DistributedModelParallel(
            module=module,
            device=device,
            env=trec_dist.ShardingEnv.from_local(world_size, default_cuda_rank),
            plan=plan,
            sharders=sharders,
            init_data_parallel=False,
        )

        return module
