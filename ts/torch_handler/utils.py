import enum


class CompilerBackend(str, enum.Enum):
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
