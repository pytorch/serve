import json
import os
from urllib import parse

import grpc
import test_gRPC_utils
import test_utils

management_data_json = "../postman/management_data.json"


def setup_module(module):
    test_utils.torchserve_cleanup()
    test_utils.start_torchserve()


def teardown_module(module):
    test_utils.torchserve_cleanup()


def __get_parsed_url(path):
    return parse.urlsplit(path)


def __get_query_params(parsed_url):
    query_params = dict(parse.parse_qsl(parsed_url.query))

    for key, value in query_params.items():
        if key in [
            "min_worker",
            "max_worker",
            "initial_workers",
            "timeout",
            "number_gpu",
            "batch_size",
            "max_batch_delay",
            "response_timeout",
            "startup_timeout",
            "limit",
            "next_page_token",
        ]:
            query_params[key] = int(query_params[key])
        if key in ["synchronous"]:
            query_params[key] = bool(query_params[key])
        if key in ["url"] and query_params[key].startswith("{{mar_path_"):
            query_params[key] = test_utils.mar_file_table[query_params[key][2:-2]]

    return query_params


def __get_path_params(parsed_url):
    path = parsed_url.path.split("/")

    if len(path) == 1:
        return {}

    path_params = {"model_name": path[1]}
    if len(path) == 3:
        path_params.update({"model_version": path[2]})

    return path_params


def __get_path_query_params(parsed_url):
    params = __get_path_params(parsed_url)
    params.update(__get_query_params(parsed_url))
    return params


def __get_register_params(parsed_url):
    return __get_query_params(parsed_url)


def __get_unregister_params(parsed_url):
    return __get_path_params(parsed_url)


def __get_scale_params(parsed_url):
    return __get_path_query_params(parsed_url)


def __get_set_default_params(parsed_url):
    return __get_path_params(parsed_url)


def __get_describe_params(parsed_url):
    return __get_path_params(parsed_url)


def __get_list_params(parsed_url):
    return __get_path_query_params(parsed_url)


def test_management_apis():
    api_mapping = {
        "register": "RegisterModel",
        "unregister": "UnregisterModel",
        "scale": "ScaleWorker",
        "set_default": "SetDefault",
        "list": "ListModels",
        "describe": "DescribeModel",
    }

    with open(os.path.join(os.path.dirname(__file__), management_data_json), "rb") as f:
        test_data = json.loads(f.read())

    for item in test_data:
        try:
            api_name = item["type"]
            api = globals()["__get_" + api_name + "_params"]
            params = api(parse.urlsplit(item["path"]))
            test_gRPC_utils.run_management_api(api_mapping[api_name], **params)
        except grpc.RpcError as e:
            if "grpc_status_code" in item:
                assert e.code().value[0] == item["grpc_status_code"]
        except ValueError as e:
            # gRPC has more stricter check on the input types hence ignoring the test case from data file
            continue
