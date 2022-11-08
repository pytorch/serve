import asyncio
import datetime
from functools import partial

import aiohttp
import torch

from torchdata.datapipes.iter import Decompressor, FileLister, FileOpener

# Batch_Size to kept 1 and letting torchserver to do batch inference
BATCH_SIZE = 1
# Torchserver model URl
MODEL_URL = "http://127.0.0.1:8080/predictions/mnist"
# Total number of inference calls to send in a single batch
BATCH_TOTAL_TASKS = 100
# Stop sending infrence request after making TOTAL_INFERENCE_CALLS
TOTAL_INFERENCE_CALLS = BATCH_TOTAL_TASKS * 2
# MNIST image size
IMAGE_SIZE = (28, 28)

# Loop through the dataset
async def iterate_dataset():
    aiohttp_client_session = aiohttp.ClientSession()
    tasks = []
    total_completed_tasks = 0
    image_dp, label_dp = MNIST()

    _, image_stream = next(iter(image_dp))
    _, label_stream = next(iter(label_dp))

    # ignoring first 16 bytes from image_stream and first 8 bytes from label_stream.
    image_stream.read(16)
    label_stream.read(8)

    # Reading one batch information from the image_stream
    n_bytes = IMAGE_SIZE[0] * IMAGE_SIZE[1] * BATCH_SIZE
    for buffer in iter(partial(image_stream.read, n_bytes), b""):
        batch = torch.frombuffer(buffer, dtype=torch.uint8).to(torch.float32, copy=True)
        # reshaping the batch
        batch = batch.reshape(1, 1, IMAGE_SIZE[0], IMAGE_SIZE[1])
        # reading the true label
        true_label = torch.frombuffer(label_stream.read(1), dtype=torch.uint8)
        if len(tasks) <= BATCH_TOTAL_TASKS:
            tasks.append(
                send_inference_request(aiohttp_client_session, batch, true_label)
            )
        else:
            # Once we've TOTAL_TASKS send all inference request in short time to allow
            # torchserver to do batch inference
            print(f"Total inference calls sent so far {len(tasks)}.")
            result = await asyncio.gather(*tasks)
            if total_completed_tasks > TOTAL_INFERENCE_CALLS:
                print(
                    f"Finished sending maximum number of inference calls {TOTAL_INFERENCE_CALLS}. Stopping..."
                )
            tasks = []
            tasks.append(
                send_inference_request(aiohttp_client_session, batch, true_label)
            )

        total_completed_tasks = total_completed_tasks + 1

    await aiohttp_client_session.close()


async def send_inference_request(aiohttp_session, image, true_label):
    """
    This functions call inference REST API with image.
    """
    data = {"image": image.tolist()}
    async with aiohttp_session.post(MODEL_URL, json=data) as resp:
        resp_text = await resp.text()
        print(
            datetime.datetime.now(),
            f"- Model prediction Class {resp_text} True Class: {true_label}",
        )
        return resp_text


def MNIST():
    label_dp = FileLister("./mnist_dataset/t10k-labels-idx1-ubyte.gz")
    label_dp = FileOpener(label_dp, mode="b")
    label_dp = Decompressor(label_dp)

    image_dp = FileLister("./mnist_dataset/t10k-images-idx3-ubyte.gz")
    image_dp = FileOpener(image_dp, mode="b")
    image_dp = Decompressor(image_dp)

    return image_dp, label_dp


if __name__ == "__main__":
    asyncio.run(iterate_dataset())
