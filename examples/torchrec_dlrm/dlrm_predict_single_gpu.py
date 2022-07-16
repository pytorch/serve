# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import torch
from dlrm_predict import DLRMModelConfig, DLRMPredictModule
from torchrec.inference.modules import PredictFactory, quantize_embeddings
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

logger: logging.Logger = logging.getLogger(__name__)

# OSS Only


class DLRMPredictSingleGPUModule(DLRMPredictModule):
    """
    nn.Module used for unsharded, single GPU, DLRM predict module. DistributedModelParallel
    (TorchRec sharding API) is not expected to wrap this module.
    """

    # TODO: Determine cleaner way to remove the copy.
    # This is needed because the server expects copy method to exist on predict module.
    def copy(self, device: torch.device):
        return self


class DLRMPredictSingleGPUFactory(PredictFactory):
    def __init__(self, model_config: DLRMModelConfig) -> None:
        self.model_config: DLRMModelConfig = model_config

    def create_predict_module(self, world_size: int) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)
        default_cuda_rank = 0
        device = torch.device("cuda", default_cuda_rank)
        torch.cuda.set_device(device)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.model_config.embedding_dim,
                num_embeddings=self.model_config.num_embeddings_per_feature[feature_idx]
                if self.model_config.num_embeddings is None
                else self.model_config.num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                self.model_config.id_list_features_keys
            )
        ]
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictSingleGPUModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            id_list_features_keys=self.model_config.id_list_features_keys,
            dense_device=device,
        )

        module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)
        # TensorRT Lowering - Use torch_tensorrt.fx (https://github.com/pytorch/TensorRT) for lowering dense module

        # Follow https://github.com/pytorch/TensorRT/blob/master/py/torch_tensorrt/fx/example/fx2trt_example.py
        # for fully detailed example on splitting and lowering a submodule.

        # Example for lowering the dense part of this DLRMPredictSingleGPUModule:

        # import torch.fx
        # import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
        # from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter

        # sample_input = {
        # "float_features": torch.ones(13),
        # "id_list_features.lengths": torch.ones(26),
        # "id_list_features.values": torch.ones(26)
        # }

        # traced = acc_tracer.trace(model, sample_input)
        # splitter = TRTSplitter(traced, sample_input)
        # split_mod = splitter()

        # Lower dense part (_run_on_acc_0, the part that can be lowered)
        # interp = TRTInterpreter(split_mod._run_on_acc_0, InputTensorSpec.from_tensors(inputs))
        # r = interp.run()
        # trt_mod = TRTModule(r.engine, r.input_names, r.output_names)
        # split_mod._run_on_acc_0 = trt_mod
        # return split_mod

        return module

    def batching_metadata(self) -> Dict[str, str]:
        return {
            "float_features": "dense",
            "id_list_features": "sparse",
        }

    def result_metadata(self) -> str:
        return "dict_of_tensor"
