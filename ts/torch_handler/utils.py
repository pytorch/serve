import enum

@add_mapping
class DynamoBackend(str, enum.Enum):
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

def add_mapping(enum_cls):
    for name in enum_cls.__MAPPING__:
        member = enum_cls.__members__[name]
        enum_cls.__MAPPING__[name] = member
    return enum_cls