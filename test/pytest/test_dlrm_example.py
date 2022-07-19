import json
import os

import pytest
import torch

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

CURR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT_DIR = os.path.normpath(os.path.join(CURR_FILE_PATH, "..", ".."))
EXAMPLE_ROOT_DIR = os.path.join(REPO_ROOT_DIR, "examples", "torchrec_dlrm")


def check_dlrm_result(res, gt):
    assert isinstance(res, dict)
    assert "default" in res
    assert res["default"] == list([pytest.approx(b) for b in gt["default"]])


def test_handler(monkeypatch, mocker):
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)

    from handler import TorchRecDLRMHandler

    handler = TorchRecDLRMHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=EXAMPLE_ROOT_DIR,
        model_file="dlrm_factory.py",
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    # Batch szie 2
    data = {
        "float_features": [
            [
                -0.8904874324798584,
                -0.5702090859413147,
                -0.13531066477298737,
                -1.8298695087432861,
                0.18680641055107117,
                -0.5029279589653015,
                0.20502178370952606,
                0.11757952719926834,
                -0.5099042654037476,
                0.29294583201408386,
                -0.17700502276420593,
                -1.6512247323989868,
                0.7418987154960632,
            ],
            [
                1.1390568017959595,
                -2.0782198905944824,
                -1.6261157989501953,
                -1.472241759300232,
                0.012050889432430267,
                -1.8349215984344482,
                1.9130446910858154,
                -0.5165563225746155,
                -1.3125498294830322,
                -1.6539082527160645,
                0.1867174506187439,
                -0.3760676383972168,
                0.42102983593940735,
            ],
        ],
        "id_list_features.lengths": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "id_list_features.values": [
            42412881,
            1107019,
            14556,
            29122,
            6732,
            9711,
            4978,
            1945,
            4614,
            82,
            1,
            2,
            2585,
            4772,
            1224,
            777,
            38,
            34,
            4928528,
            18233101,
            831084,
            438459,
            38202,
            251387,
            5,
            3,
            99,
            555,
            5694,
            2608,
            46,
            77,
            1,
            1,
            373,
            935,
            12,
            12,
            20073534,
            26813373,
            478696,
            1519603,
            36774469,
            4455968,
            10594,
            383035,
            7647,
            10260,
            4,
            9,
            6,
            11,
        ],
    }

    x = mocker.Mock(get=lambda x: json.dumps(data))

    x = handler.preprocess([x])
    x = handler.inference(x)
    x = handler.postprocess(x)

    check_dlrm_result(
        json.loads(x), {"default": [-0.0376037061214447, -0.037387914955616]}
    )

    # Batch size 1

    data = {
        "float_features": [
            [
                0.015471375547349453,
                0.20300938189029694,
                -1.3055355548858643,
                -0.7300364971160889,
                0.06900127977132797,
                0.5859290957450867,
                1.3041515350341797,
                -0.6238508820533752,
                1.4023090600967407,
                -1.16234290599823,
                -0.19111162424087524,
                0.8572622537612915,
                -0.2385675013065338,
            ]
        ],
        "id_list_features.lengths": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "id_list_features.values": [
            45821459,
            29807,
            12033,
            3092,
            4119,
            2,
            6527,
            194,
            34,
            8190138,
            397269,
            204955,
            4,
            914,
            3472,
            45,
            3,
            809,
            1,
            3026630,
            9850984,
            14193736,
            4706,
            7013,
            71,
            6,
        ],
    }

    x = mocker.Mock(get=lambda x: json.dumps(data))

    x = handler.preprocess([x])
    x = handler.inference(x)
    x = handler.postprocess(x)

    check_dlrm_result(json.loads(x), {"default": [-0.037986066192388535]})
