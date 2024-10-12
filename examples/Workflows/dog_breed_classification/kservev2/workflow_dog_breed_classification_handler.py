import json


def pre_processing(data, context):
    """
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    {
        "id": "f0222600-353f-47df-8d9d-c96d96fa894e",
        "inputs": [{
            "name": "input-0",
            "shape": [37],
            "datatype": "INT64",
            "data": [66, 108, 111, 111, 109]
        }]
    }
    """
    if data is None:
        return data
    # b64_data.append(base64.b64encode(data).decode())
    for row in data:
        input_data = row.get("data") or row.get("body")
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        input_data = json.loads(input_data.decode("utf8"))
        setattr(context, "request_ids", input_data["id"])
        raw_data_list = [item["data"][0] for item in input_data["inputs"]]
        # b64_data.append(base64.b64encode(input_data).decode())
    return raw_data_list


def post_processing(data, context):
    """
    {
      "id": "f0222600-353f-47df-8d9d-c96d96fa894e",
      "model_name": "bert",
      "model_version": "1",
      "outputs": [{
        "name": "input-0",
        "shape": [1],
        "datatype": "INT64",
        "data": [2]
      }]
    }
    """
    if data is None:
        return data
    b64_data = []
    id = context.get_request_id(0)
    model_name = context.model_name
    for row in data:
        input_data = row.get("data") or row.get("body")
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        input_data = json.loads(input_data.decode("utf8"))
        b64_data.append(
            {"name": "input-0", "shape": [-1], "data": input_data, "datatype": "BYTES"}
        )
        # b64_data.append(base64.b64encode(input_data).decode())
    response = {"id": id, "model_name": model_name, "outputs": b64_data}
    return response
