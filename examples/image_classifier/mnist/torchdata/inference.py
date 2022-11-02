import asyncio
import datetime

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# Batch_Size to kept 1 and letting torchserver to do batch inference
BATCH_SIZE = 1
# Torchserver model URl
MODEL_URL = "http://127.0.0.1:8080/predictions/mnist"
# Total number of inference calls to send in a single batch
BATCH_TOTAL_TASKS = 100
# Stop sending infrence request after making TOTAL_INFERENCE_CALLS
TOTAL_INFERENCE_CALLS = BATCH_TOTAL_TASKS * 2

#transform the images to tensor. Normalize the images as well.
image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

# Load the dataset and set train flag to False.
testset = datasets.MNIST('./MNIST_dataset', download=True, train=False, transform=image_transform)

# Creating the dataloader.
inference_dataset = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Loop through the dataset
async def iterate_dataset():
    aiohttp_client_session = aiohttp.ClientSession()
    tasks = []
    total_completed_tasks = 0
    for batch, true_label in iter(inference_dataset):
        if len(tasks) <= BATCH_TOTAL_TASKS:
            tasks.append(
                send_inference_request(
                    aiohttp_client_session, batch, true_label
                )
            )
        else:
            # Once we've TOTAL_TASKS send all inference request in short time to allow
            # torchserver to do batch inference
            print(
                f"Total inference calls sent so far {len(tasks)}."
            )
            result = await asyncio.gather(*tasks)
            if total_completed_tasks > TOTAL_INFERENCE_CALLS:
                print(
                    f"Finished sending maximum number of inference calls {TOTAL_INFERENCE_CALLS}. Stopping..."
                )
            tasks = []
            tasks.append(
                send_inference_request(
                    aiohttp_client_session, batch, true_label
                )
            )

        total_completed_tasks = total_completed_tasks + 1

    await aiohttp_client_session.close()

async def send_inference_request(aiohttp_session, image, true_label):
    """
    This functions call inference REST API with image.
    """
    data = {'image': image.tolist()}
    async with aiohttp_session.post(MODEL_URL, json=data) as resp:
        resp_text = await resp.text()
        print(datetime.datetime.now(), f"- Model prediction Class {resp_text} Actual Class: {true_label}")
        return resp_text

if __name__ == "__main__":
    asyncio.run(iterate_dataset())
