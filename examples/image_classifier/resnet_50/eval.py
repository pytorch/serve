import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

cudnn.benchmark = True


def benchmark(
    model, input_shape=(1024, 1, 224, 224), dtype="fp32", nwarmup=50, nruns=10000
):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == "fp16":
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


def rn50_preprocess():
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess


# img = Image.open("../kitten.jpg")
# preprocess = rn50_preprocess()
# input_tensor = preprocess(img)
# input_batch = input_tensor.unsqueeze(0)
#
# model = torch.jit.load("trt_model_fp32.ts")
#
# input_batch = input_batch.to('cuda')
# model.to('cuda')
#
# with torch.no_grad():
#        output = model(input_batch)
#        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#        sm_output = torch.nn.functional.softmax(output[0], dim=0)

# Model benchmark without Torch-TensorRT

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

model.eval().to("cuda")
print("###################################################")
print("Benchmarking ResNet50 model")
benchmark(model, input_shape=(64, 3, 224, 224), nruns=100)

model = torch.jit.load("trt_model_fp32.ts")
model.eval().to("cuda")
print("###################################################")
print("Benchmarking ResNet50 Tensor RT FP32 model")
benchmark(model, input_shape=(64, 3, 224, 224), nruns=100)

model = torch.jit.load("trt_model_fp16.ts")
model.eval().to("cuda")
print("###################################################")
print("Benchmarking ResNet50 Tensor RT FP16 model")
benchmark(model, input_shape=(64, 3, 224, 224), nruns=100)
