import json
from PIL import Image
from torchvision import transforms

image = Image.open('0.png')  # PIL's JpegImageFile format (size=(W,H))
print(image.size)  # (W，H）
tran = transforms.ToTensor(
)  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
img_tensor = tran(image)
print(img_tensor.shape)
print(img_tensor.dtype)
print(img_tensor)  # (C,H, W), channel order (R, G, B)
request = {
    "inputs": [{
        "name": "input-0",
        "shape": img_tensor.shape,
        "datatype": "FP32",
        "data": [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]
    }]
}
with open('mnist_v21.json', 'w') as outfile:
    json.dump(request, outfile, indent=4, sort_keys=True)
