import copy
import pickle
from collections import namedtuple
from itertools import chain
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

ShmInfo = namedtuple("ShmInfo", ["shape", "dtype", "shm"])


def tensor_to_shm(name: str, t: torch.Tensor) -> ShmInfo:
    t_np = t.detach().numpy()
    d_size = np.dtype(t_np.dtype).itemsize * np.prod(t_np.shape)
    try:
        shm = SharedMemory(create=True, size=d_size, name=name)
    except FileNotFoundError:
        shm = SharedMemory(create=False, size=d_size, name=name)

    t_np_shared = np.ndarray(shape=t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    t_np_shared[:] = t_np[:]

    shm_info = ShmInfo(t_np_shared.shape, t_np_shared.dtype, shm)

    return shm_info


def strip_model(name: str, model: nn.Module) -> Tuple:
    arrays = []
    shm = []

    for m_name, m in model.named_modules():
        params = {
            n: tensor_to_shm(f"{name}_{m_name}_{n}", t)
            for n, t in m.named_parameters(recurse=False)
        }
        buffers = {
            n: tensor_to_shm(f"{name}_{m_name}_{n}", t)
            for n, t in m.named_buffers(recurse=False)
        }
        shm += [info.shm for _, info in chain(params.items(), buffers.items())]
        arrays.append({"params": params, "buffers": buffers})

    model_empty = copy.deepcopy(model)
    for m in model_empty.modules():
        for n, _ in chain(m.named_parameters(), m.named_buffers()):
            setattr(m, n, None)

    return model_empty, arrays, shm


def shm_info_to_np(info: ShmInfo) -> np.ndarray:
    return np.ndarray(
        shape=info.shape,
        dtype=info.dtype,
        buffer=info.shm.buf,
    )


def restore_model(model: nn.Module, arrays: Tuple) -> nn.Module:
    for (_, m), pb in zip(model.named_modules(), arrays):
        for name, info in pb["params"].items():
            m.register_parameter(
                name, torch.nn.Parameter(torch.as_tensor(shm_info_to_np(info)))
            )
        for name, info in pb["buffers"].items():
            m.register_buffer(name, torch.as_tensor(shm_info_to_np(info)))
    model.train(False)


class SharedMemoryModelStore(object):
    def __init__(self, store_name: str):
        self.store_name = store_name
        self.store = dist.FileStore(str(Path("/tmp") / store_name), -1)
        self.shm = []
        self.shm_copies = []
        self.arrays = []

    def __del__(self):
        for s in self.shm_copies:
            s.close()

        for s in self.shm:
            s.close()
            s.unlink()

    def set(self, model_name: str, model: nn.Module) -> nn.Module:
        # assert self.store.get(model_name) is None

        m_empty, arrays, shm = strip_model(f"{self.store_name}_{model_name}", model)
        self.shm += shm
        self.arrays += arrays

        self.store.set(model_name, pickle.dumps((m_empty, arrays)))

        restore_model(m_empty, arrays)

        return m_empty

    def get(self, model_name: str) -> nn.Module:

        m_empty, arrays = pickle.loads(self.store.get(model_name))

        # We need to keep the shared memory objects in memory to use the buf memory view
        shm = []
        for a in arrays:
            shm += [
                info.shm for _, info in chain(a["params"].items(), a["buffers"].items())
            ]
        self.shm_copies += shm

        restore_model(m_empty, arrays)

        return m_empty
