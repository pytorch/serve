import json


def post_processing(data, context):
    """
    Changes the output keys obtained from the individual model
    to be more appropriate for the workflow output
    """
    if data:
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")
        data = json.loads(data)
        data["german_translation"] = data.pop("input")
        data["english_re_translation"] = data.pop("english_output")
        return [data]


def prep_intermediate_input(data, context):
    """
    Extracts only the translated text from the output of the first model
    and converts it into the string that is expected by the second model
    """
    if data:
        if isinstance(data, list):
            data = data[0]
        data = data.get("data") or data.get("body")
        data = json.loads(data)
        data = data["german_output"]
        return [data]
