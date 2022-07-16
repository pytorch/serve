import argparse
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset


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

    dl = DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=2,
            hash_size=len(args.num_embeddings_per_feature),
            hash_sizes=list(map(int, args.num_embeddings_per_feature.split(","))),
            manual_seed=None,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=True,
        num_workers=0,
    )

    model = torch.load("dlrm.pt")

    device = torch.device("cuda", 0)

    batch = next(iter(dl))

    input_data = {
        "float_features": batch.dense_features.to(device),
        "id_list_features.lengths": batch.sparse_features.lengths().to(device),
        "id_list_features.values": batch.sparse_features.values().to(device),
    }

    print(input_data)

    print(model(input_data))


if __name__ == "__main__":
    main(sys.argv[1:])
