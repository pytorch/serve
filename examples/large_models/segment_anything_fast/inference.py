import base64
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from pycocotools import mask as coco_mask

url = "http://localhost:8080/predictions/sam-fast"
image_path = "./kitten.jpg"


def show_anns(anns):
    if len(anns) == 0:
        return
    for i in range(len(anns)):
        anns[i]["segmentation"] = coco_mask.decode(anns[i]["segmentation"])
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"].astype(bool)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# Send Inference request to TorchServe
file = {"body": open(image_path, "rb")}
res = requests.post(url, files=file)

# Decode the Base64 encoded data (if needed)
decoded_data = base64.b64decode(res.text)

# Deserialize the data using Pickle
masks = pickle.loads(decoded_data)


# Plot the segmentation mask on the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.tight_layout()
plt.savefig("kitten_mask_fast.png", format="png")
