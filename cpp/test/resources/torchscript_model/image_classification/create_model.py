import importlib.machinery
import importlib.util
import pathlib
from argparse import ArgumentParser

import PIL.Image as Image
import requests
import torch
from torchvision import transforms

IMG_URL = "https://pytorch.org/vision/stable/_images/sphx_glr_plot_scripted_tensor_transforms_001.png"


def main(args):
    path = (
        pathlib.Path(__file__).parents[5]
        / "examples"
        / "image_classifier"
        / "resnet_18"
        / "model.py"
    )

    assert path.exists()
    loader = importlib.machinery.SourceFileLoader("model", str(path))
    spec = importlib.util.spec_from_loader("model", loader)
    model = importlib.util.module_from_spec(spec)
    loader.exec_module(model)

    resnet_18 = model.ImageClassifier()

    combined = torch.nn.Sequential(
        transforms.Resize([256, 256]),
        transforms.CenterCrop([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        resnet_18,
    )

    combined = torch.jit.script(combined)
    torch.jit.save(combined, args.model_file)

    if args.data_file:
        response = requests.get(IMG_URL, stream=True)
        img = Image.open(response.raw)

        img = transforms.functional.to_tensor(img)[:3, ...]

        torch.save(img, args.data_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-file", default="model.pt", type=str)
    parser.add_argument("--data-file", default=None, type=str)
    main(parser.parse_args())
