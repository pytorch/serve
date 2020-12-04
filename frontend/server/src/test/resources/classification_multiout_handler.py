
def pre_processing(data, context):
    if data:
        image = data[0].get("data") or data[0].get("body")
        return [image]
    else:
        return None


def post_processing(data, context):
    if data:
        image = data[0].get("data") or data[0].get("body")
        return [image]
    else:
        return None
