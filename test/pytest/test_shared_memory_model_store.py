import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from ts.shared_memory_model_store import SharedMemoryModelStore

MODEL_NAME = "some_model"
MODEL_VERSION = "v1"


def test_shm_model_store_simple():
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
    model2 = shm_model_store.get(MODEL_NAME)

    assert all(
        torch.equal(a[0], b[0]) for a, b in zip(model.parameters(), model2.parameters())
    )

    input_1 = torch.rand(
        10,
    )
    output_1 = model(input_1)
    assert torch.equal(output_1, model2(input_1))

    with torch.no_grad():
        for p in model.parameters():
            p[:] = -p[:]

    output_2 = model(input_1)
    assert not torch.equal(output_1, output_2)

    assert all(
        torch.equal(a[0], b[0]) for a, b in zip(model.parameters(), model2.parameters())
    )

    input_2 = torch.rand(
        10,
    )
    assert torch.equal(model(input_2), model2(input_2))


def test_shm_model_store_mp():
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

    p1_data = mp.Value("d")
    terminate = mp.Event()
    setup_complete = mp.Event()
    p1 = mp.Process(target=func, args=(p1_data, terminate, setup_complete))
    p1.daemon = True

    p1.start()
    setup_complete.wait()

    model_store = SharedMemoryModelStore(f"{MODEL_NAME}_{MODEL_VERSION}")
    model = model_store.get(MODEL_NAME)

    assert model is not None

    value = model(-torch.as_tensor(np.ones((10,), dtype=np.float32))).item()

    terminate.set()
    p1.join()

    assert p1_data.value == value
