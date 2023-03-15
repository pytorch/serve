import base64
import json
from typing import Union

from inference_pb2 import PredictionResponse, PredictionsRequest
from kserve.errors import InvalidInput
from kserve.protocol.grpc.grpc_predict_v2_pb2 import (
    InferTensorContents,
    ModelInferRequest,
)
from kserve.protocol.infer_type import InferOutput, InferRequest, InferResponse


def get_content(datatype: str, data: InferTensorContents):
    if datatype == "BOOL":
        return list(data.bool_contents)
    elif datatype in ["UINT8", "UINT16", "UINT32"]:
        return list(data.uint_contents)
    elif datatype == "UINT64":
        return list(data.uint64_contents)
    elif datatype in ["INT8", "INT16", "INT32"]:
        return list(data.int_contents)
    elif datatype == "INT64":
        return list(data.int64_contents)
    elif datatype == "FP32":
        return list(data.fp32_contents)
    elif datatype == "FP64":
        return list(data.fp64_contents)
    elif datatype == "BYTES":
        return [base64.b64encode(data.bytes_contents[0]).decode("utf-8")]
    else:
        raise InvalidInput("invalid content type")


def to_ts_grpc(data: Union[ModelInferRequest, InferRequest]) -> PredictionsRequest:
    """Converts the InferRequest object to Torchserve gRPC PredictionsRequest message"""
    if isinstance(data, InferRequest):
        data = data.to_grpc()
    infer_request = {}
    model_name = data.model_name
    infer_inputs = [
        dict(
            name=input_tensor.name,
            shape=list(input_tensor.shape),
            datatype=input_tensor.datatype,
            data=get_content(input_tensor.datatype, input_tensor.contents),
        )
        for input_tensor in data.inputs
    ]
    infer_request["id"] = data.id
    infer_request["inputs"] = infer_inputs
    ts_grpc_input = {"data": json.dumps(infer_request).encode("utf-8")}
    return PredictionsRequest(model_name=model_name, input=ts_grpc_input)


def from_ts_grpc(data: PredictionResponse) -> InferResponse:
    """Converts the Torchserve gRPC PredictionResponse object to InferResponse message"""
    decoded_data = json.loads(data.prediction.decode("utf-8"))
    infer_outputs = [
        InferOutput(
            name=output["name"],
            shape=list(output["shape"]),
            datatype=output["datatype"],
            data=output["data"],
        )
        for output in decoded_data["outputs"]
    ]
    response_id = decoded_data.get("id")
    infer_response = InferResponse(
        model_name=decoded_data["model_name"],
        response_id=response_id,
        infer_outputs=infer_outputs,
        from_grpc=True,
    )
    return infer_response.to_grpc()
