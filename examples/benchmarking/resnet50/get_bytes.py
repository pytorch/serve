import io
import time

import torch
from PIL import Image
from torchvision import transforms

im_file = "../../image_classifier/kitten.jpg"

# with open("../kitten.jpg", "rb") as img_file:
#    my_string = base64.b64encode(img_file.read())
#
# data = {"data": my_string.decode("utf-8")}
#
# with open("data.json", "w") as outfile:
#    json.dump(data, outfile)
#
#
# with open(image, "rb") as image:
#  f = image.read()
#  im_bytes = bytearray(f)
# with open("data.txt", "wb") as binary_file:
#    # Write bytes to file
#    binary_file.write(im_bytes)
#
#
# image = Image.open(io.BytesIO(im_bytes))

image = Image.open(im_file)

image_processing = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print(image.size)
start = time.time()
image = image_processing(image)
end = time.time()
print("Time taken ", end - start)
print(image.shape)


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# im_bytes = image_to_byte_array(image)
#
# with open("data.txt", "wb") as binary_file:
#    # Write bytes to file
#    binary_file.write(im_bytes)

torch.save(image, "data.txt")

test = torch.load("data.txt")
print(test.shape)
