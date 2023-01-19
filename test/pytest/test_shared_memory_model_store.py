from itertools import chain

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ts.shared_memory_model_store import SharedMemoryModelStore

MODEL_NAME = "some_model"
MODEL_VERSION = "v1"


def modules_are_equal(model1: nn.Module, model2: nn.Module) -> bool:
    params = zip(model1.parameters(), model2.parameters())
    buffer = zip(model1.buffers(), model2.buffers())
    return all(torch.equal(a[0], b[0]) for a, b in chain(params, buffer))


def test_shm_model_store_simple():
    # This test covers the basic functionallity within thew same process
    torch.manual_seed(42)

    shm_model_store = SharedMemoryModelStore(f"{MODEL_NAME}_{MODEL_VERSION}")

    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.ReLU(),
    )
    model.train(False)

    # Round trip through store
    model = shm_model_store.set(MODEL_NAME, model)
    model2 = shm_model_store.get(MODEL_NAME)

    # models should be equivalent
    assert modules_are_equal(model, model2)

    input_1 = torch.rand(
        10,
    )
    output_1 = model(input_1)
    assert torch.equal(output_1, model2(input_1))

    with torch.no_grad():
        for p in model.parameters():
            p[:] = -p[:]

    # Modification should lead to different result
    output_2 = model(input_1)
    assert not torch.equal(output_1, output_2)

    # after modification models should still be equivalent
    assert modules_are_equal(model, model2)

    input_2 = torch.rand(
        10,
    )
    assert torch.equal(model(input_2), model2(input_2))


def test_shm_model_store_mp():
    # This test runs a seperate process and shares the model between main and subprocess
    def func(data, terminate, setup_complete):
        torch.manual_seed(42)
        shm_model_store = SharedMemoryModelStore(f"{MODEL_NAME}_{MODEL_VERSION}")

        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.ReLU(),
        )
        model.train(False)

        model = shm_model_store.set(MODEL_NAME, model)

        setup_complete.set()

        data.value = model(-torch.as_tensor(np.ones((10,), dtype=np.float32)))

        terminate.wait()

    # Setup process and communication primitives
    result = mp.Value("d")
    terminate = mp.Event()
    setup_complete = mp.Event()

    subprocess = mp.Process(target=func, args=(result, terminate, setup_complete))
    subprocess.daemon = True

    subprocess.start()
    setup_complete.wait()

    # Get model created in subprocess
    model_store = SharedMemoryModelStore(f"{MODEL_NAME}_{MODEL_VERSION}")
    model = model_store.get(MODEL_NAME)

    assert model is not None

    value = model(-torch.as_tensor(np.ones((10,), dtype=np.float32))).item()

    # Signal subprocess to terminate itself and join
    terminate.set()
    subprocess.join()

    # Model should have the same output
    assert result.value == value
