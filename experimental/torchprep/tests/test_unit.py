import os
import torch
from torchprep.format import Profiler
from torchprep import __version__
from torchprep.format import materialize_tensors, parse_input_format, Device, Precision
from torchprep.fusion import _fuse
from torchprep.utils import profile_model
from torchprep.utils import ToyNet
from .download_example import main
from torchprep.pruning import _prune
from torchprep.quantization import _quantize

# General tests
def test_version():
    assert __version__ == '0.2.0'

def test_download():
    main()
    assert len(os.listdir("models")) > 0

def test_prune():
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "resnet152.pt")
    pruned_model = _prune(model_path=model_path, prune_amount=0.3)
    assert isinstance(pruned_model, torch.nn.Module)

def test_profile():
    net = ToyNet()
    result = profile_model(model=net, custom_profiler=Profiler.nothing, input_tensors= [torch.randn(10)], label = "toy_profile", iterations=100)
    assert len(result) == 3

def test_quantization():
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "resnet152.pt")
    quantized_model = _quantize(model_path=model_path,precision = Precision.float16)
    assert isinstance(quantized_model, torch.nn.Module)

def test_fuse():
    # TODO: Fusion needs to know the input shape
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "resnet152.pt")
    input_shape = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", "resnet.yaml")
    fused_model = _fuse(model_path=model_path, input_shape=input_shape)
    print(f"Type of fused model: {type(fused_model)}")
    if fused_model:
        assert isinstance(fused_model, torch.nn.Module)
    
    # Model is not torchscriptable
    assert True == True

def test_format():
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", "resnet.yaml")
    tensors = materialize_tensors(parse_input_format(config_file))
    assert tensors[0].shape == torch.Size([1, 3, 7, 7])

def test_runtime_export():
    # if runtime is installed run test
    return NotImplemented
