import inference_pb2
import numpy
from kserve.protocol.infer_type import InferRequest


def to_ts_grpc(data: InferRequest) -> inference_pb2.PredictionsRequest:
    """Converts the InferRequest object to Torchserve gRPC PredictionsRequest message"""
    infer_inputs = []
    model_name = data.model_name
    for infer_input in data.inputs:
        infer_input_dict = {
            "name": infer_input.name,
            "shape": infer_input.shape,
            "datatype": infer_input.datatype,
        }
        if isinstance(infer_input.data, numpy.ndarray):
            infer_input.set_data_from_numpy(infer_input.data, binary_data=False)
            infer_input_dict["data"] = infer_input.data
        else:
            infer_input_dict["data"] = infer_input.data
        infer_inputs.append(infer_input.data)
    input_data = {"data": infer_inputs[0][0]}
    # infer_request = {}
    # infer_request["inputs"] = infer_inputs
    return inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)

    # infer_inputs = []
    # model_name = data.model_name
    # for infer_input in data.inputs:
    #     if isinstance(infer_input.data, numpy.ndarray):
    #         infer_input.set_data_from_numpy(infer_input.data, binary_data=True)
    #     infer_input_dict = {}
    #     if not isinstance(infer_input.data, List):
    #         raise InvalidInput("input data is not a List")
    #     infer_input_dict["data"] = infer_input.data
    #     infer_inputs.append(infer_input_dict)
    # return inference_pb2.PredictionsRequest(model_name= model_name, inputs=infer_inputs)
