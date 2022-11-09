import enum

class DynamoBackend(str, enum.Enum):
    """
    Represents a dynamo backend (see https://github.com/pytorch/torchdynamo).
    Values:
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **NVFUSER** -- nvFuser with TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_NVFUSER** -- nvFuser with AotAutograd. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read
          more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).
    """
    EAGER = "eager"
    AOT_EAGER = "aot_eager"
    INDUCTOR = "inductor"
    NVFUSER = "nvfuser"
    AOT_NVFUSER = "aot_nvfuser"
    AOT_CUDAGRAPHS = "aot_cudagraphs"
    OFI = "ofi"
    FX2TRT = "fx2trt"
    ONNXRT = "onnxrt"
    IPEX = "ipex"