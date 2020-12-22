
def pre_processing(data, context):
    if data:
        image = data[0].get("data") or data[0].get("body")
        return [image]
    else:
        return None


def aggregate_func(data, context):
    if data:
        resnet_result = data[0].get("resnet")
        squeezenet_result = data[0].get("squeezenet")
        return [resnet_result]
    else:
        return None
