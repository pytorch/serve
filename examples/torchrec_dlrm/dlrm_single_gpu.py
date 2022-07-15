#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch

from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.inference.model_packager import load_pickle_config
from torchrec.inference.modules import quantize_embeddings
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class DLRMModelConfig:
    dense_arch_layer_sizes: List[int]
    dense_in_features: int
    embedding_dim: int
    id_list_features_keys: List[str]
    num_embeddings_per_feature: List[int]
    num_embeddings: int
    over_arch_layer_sizes: List[int]


def create_predict_module(model_config: DLRMModelConfig) -> torch.nn.Module:
    default_cuda_rank = 0
    device = torch.device("cuda", default_cuda_rank)
    torch.cuda.set_device(device)

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=model_config.embedding_dim,
            num_embeddings=model_config.num_embeddings_per_feature[feature_idx]
            if model_config.num_embeddings is None
            else model_config.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(
            model_config.id_list_features_keys
        )
    ]
    ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

    module = DLRM(
        embedding_bag_collection=ebc,
        dense_in_features=model_config.dense_in_features,
        dense_arch_layer_sizes=model_config.dense_arch_layer_sizes,
        over_arch_layer_sizes=model_config.over_arch_layer_sizes,
        # id_list_features_keys=model_config.id_list_features_keys,
        dense_device=device,
    )

    module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)
    
    return module


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm model packager")
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,"
        "10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--sparse_feature_names",
        type=str,
        default=",".join(DEFAULT_CAT_NAMES),
        help="Comma separated names of the sparse features.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--num_dense_features",
        type=int,
        default=len(DEFAULT_INT_NAMES),
        help="Number of dense features.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path of model package.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    Use torch.package to package the torchrec DLRM Model.

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)

    model_config = DLRMModelConfig(
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        dense_in_features=args.num_dense_features,
        embedding_dim=args.embedding_dim,
        id_list_features_keys=args.sparse_feature_names.split(","),
        num_embeddings_per_feature=list(
            map(int, args.num_embeddings_per_feature.split(","))
        ),
        num_embeddings=args.num_embeddings,
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
    )

    model = create_predict_module(model_config)

    torch.save(model, "dlrm.pt")
    

if __name__ == "__main__":
    main(sys.argv[1:])
