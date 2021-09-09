import json
import base64

def pre_processing(data, context):
    '''
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    '''
    if data is None:
        return data
    b64_data = []
    for row in data:
        input_data = row.get("data") or row.get("body")
        # Base64 encode the image to avoid the framework throwing
        # non json encodable errors
        b64_data.append(base64.b64encode(input_data).decode())
    return b64_data
