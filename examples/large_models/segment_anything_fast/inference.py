import base64
import json
import pickle
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

url = "http://localhost:8080/predictions/sam-fast"
image_path = "./kitten.jpg"


class NumpyArrayDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        decoded_data = super(NumpyArrayDecoder, self).decode(s, **kwargs)
        return self._decode_numpy_arrays(decoded_data)

    def _decode_numpy_arrays(self, data):
        if isinstance(data, list):
            return [self._decode_numpy_arrays(item) for item in data]
        elif isinstance(data, dict):
            return {
                key: self._decode_numpy_arrays(value) for key, value in data.items()
            }
        elif isinstance(data, str) and data.startswith("__nparray__"):
            array_data = json.loads(data[11:])
            return np.array(array_data)
        else:
            return data


def show_anns(anns):
    if len(anns) == 0:
        return
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
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


file = {"body": open(image_path, "rb")}

res = requests.post(url, files=file)

# Decode the Base64 encoded data (if needed)
decoded_data = base64.b64decode(res.text)

# Deserialize the data using Pickle
decompressed_data = pickle.loads(decoded_data)

# Decompress the decompressed data (if needed)
decompressed_string = zlib.decompress(decompressed_data).decode("utf-8")

# convert the string into numpy array
masks = json.loads(decompressed_string, cls=NumpyArrayDecoder)["data"]
masks = np.array(masks)


# Plot the segmentation mask on the image
plt.figure(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.tight_layout()
plt.savefig("kitten_mask_fast.png", format="png")
