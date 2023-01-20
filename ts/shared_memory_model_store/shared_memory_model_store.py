"""
SharedMemoryModelStore is a class to exchange module between processes by storing the weights on shared memory
"""

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

# Shared memory info contains a pointer to shared memory
# as well as type and information necessary to recreate the tensor from it
ShmInfo = namedtuple("ShmInfo", ["shape", "dtype", "shm"])


def tensor_to_shm(name: str, t: torch.Tensor) -> ShmInfo:
    """
    This method moves the data of a tensor into shared memory
    The returned information can be used to recreate the tnesor

    :param name: Unique identifier
    :param t: a tensor whole data will be moves to shared memory
    :return: info about location, shape and type of data
    """

    t_np = t.detach().numpy()
    d_size = np.dtype(t_np.dtype).itemsize * np.prod(t_np.shape)

    shm = SharedMemory(create=True, size=d_size, name=name)

    t_np_shared = np.ndarray(shape=t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    t_np_shared[:] = t_np[:]

    shm_info = ShmInfo(t_np_shared.shape, t_np_shared.dtype, shm)

    return shm_info


def strip_model(name: str, model: nn.Module) -> Tuple:
    """
    This method moves all parameters and buffers to shared memory
    and return them separately to a module with stripped parameters and buffers

    :param name: Unique identifier
    :param model: module to be stripped of parameters
    :return: empty module, the parameters and buffer in shared memory
    """

    # Move all parameters and buffers to shared memory
    arrays = []
    for m_name, m in model.named_modules():
        params = {
            n: tensor_to_shm(f"{name}_{m_name}_{n}", t)
            for n, t in m.named_parameters(recurse=False)
        }
        buffers = {
            n: tensor_to_shm(f"{name}_{m_name}_{n}", t)
            for n, t in m.named_buffers(recurse=False)
        }
        arrays.append({"params": params, "buffers": buffers})

    # Create empty hull of model
    model_empty = copy.deepcopy(model)
    for m in model_empty.modules():
        for n, _ in chain(m.named_parameters(), m.named_buffers()):
            setattr(m, n, None)

    return model_empty, arrays


def shm_info_to_np(info: ShmInfo) -> np.ndarray:
    return np.ndarray(
        shape=info.shape,
        dtype=info.dtype,
        buffer=info.shm.buf,
    )


def restore_model(model: nn.Module, arrays: Tuple) -> nn.Module:
    """
    This method fills the parameter and buffers with tensors pointing to shared memory

    :param model: Module stripped of parameters
    :param arrays: a list with information about the weights in shared memory
    :return: None
    """
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
        self.store_path = Path("/tmp") / store_name
        self.store = dist.FileStore(self.store_path.as_posix(), -1)
        ret = self.store.add("num_worker", 1)
        # First worker self declares itself to master
        self.is_master = ret == 1

        # Pointers to shared memory created within the store
        self.shm = []
        self.shm_copies = []

    def __del__(self):
        for s in self.shm_copies:
            s.close()

        for s in self.shm:
            s.close()
            s.unlink()

        self.store.add("num_worker", -1)

        if self.is_master:
            self.store_path.unlink()

    def set(self, model_name: str, model: nn.Module) -> nn.Module:
        m_empty, arrays = strip_model(f"{self.store_name}_{model_name}", model)

        # Keep records of shared memory to keep it available until deletion of store
        for a in arrays:
            self.shm += [
                info.shm for _, info in chain(a["params"].items(), a["buffers"].items())
            ]

        # Move empty model and pointers to shared memory the common store
        self.store.set(model_name, pickle.dumps((m_empty, arrays)))

        # Recreate model with weights in shared memory
        restore_model(m_empty, arrays)

        return m_empty

    def get(self, model_name: str) -> nn.Module:
        m_empty, arrays = pickle.loads(self.store.get(model_name))

        # We need to keep the shared memory objects in memory to use the buf memory view
        for a in arrays:
            self.shm_copies += [
                info.shm for _, info in chain(a["params"].items(), a["buffers"].items())
            ]

        restore_model(m_empty, arrays)

        return m_empty
