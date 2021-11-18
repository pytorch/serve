import json


def pre_processing(data, context):
    """
    Empty node as a starting node since the DAG doesn't support multiple start nodes
    """
    if data:
        text = data[0].get("data") or data[0].get("body")
        return [text]
    return None


def aggregate_func(data, context):
    """
    Changes the output keys obtained from the individual model
    to be more appropriate for the workflow output
    """
    if data:
        en_de_result = json.loads(data[0].get("nmt_en_de"))
        en_fr_result = json.loads(data[0].get("nmt_en_fr"))
        response = {
            "english_input": en_de_result["input"],
            "german_translation": en_de_result["german_output"],
            "french_translation": en_fr_result["french_output"],
        }
        return [response]
