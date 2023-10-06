# coding=utf-8


"""
On The Fly Codec tester
"""

import struct
from builtins import bytes
from collections import namedtuple

import pytest

import ts.protocol.otf_message_handler as codec


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple("Patches", ["socket"])
    mock_patch = Patches(mocker.patch("socket.socket"))
    mock_patch.socket.recv.return_value = b"1"
    return mock_patch


# noinspection PyClassHasNoInit
class TestOtfCodecHandler:
    def test_retrieve_msg_unknown(self, socket_patches):
        socket_patches.socket.recv.side_effect = [b"U", b"\x00\x00\x00\x03"]
        with pytest.raises(ValueError, match=r"Invalid command: .*"):
            codec.retrieve_msg(socket_patches.socket)

    def test_retrieve_msg_load_gpu(self, socket_patches):
        expected = {
            "modelName": b"model_name",
            "modelPath": b"model_path",
            "batchSize": 1,
            "handler": b"handler",
            "gpu": 1,
            "envelope": b"envelope",
            "limitMaxImagePixels": True,
        }

        socket_patches.socket.recv.side_effect = [
            b"L",
            b"\x00\x00\x00\x0a",
            b"model_name",
            b"\x00\x00\x00\x0a",
            b"model_path",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x07",
            b"handler",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x08",
            b"envelope",
            b"\x01",
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"L"
        assert ret == expected

    def test_retrieve_msg_load_no_gpu(self, socket_patches):
        expected = {
            "modelName": b"model_name",
            "modelPath": b"model_path",
            "batchSize": 1,
            "handler": b"handler",
            "envelope": b"envelope",
            "limitMaxImagePixels": True,
        }

        socket_patches.socket.recv.side_effect = [
            b"L",
            b"\x00\x00\x00\x0a",
            b"model_name",
            b"\x00\x00\x00\x0a",
            b"model_path",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x07",
            b"handler",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x08",
            b"envelope",
            b"\x01",
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"L"
        assert ret == expected

    def test_retrieve_msg_predict(self, socket_patches):
        expected = [
            {
                "requestId": b"request_id",
                "headers": [],
                "parameters": [
                    {
                        "name": "input_name",
                        "contentType": "application/json",
                        "value": {"data": "value"},
                    }
                ],
            }
        ]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a",
            b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a",
            b"input_name",
            b"\x00\x00\x00\x0F",
            b"application/json",
            b"\x00\x00\x00\x0F",
            b'{"data":"value"}',
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF",  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"I"
        assert ret == expected

    def test_retrieve_msg_predict_text(self, socket_patches):
        expected = [
            {
                "requestId": b"request_id",
                "headers": [],
                "parameters": [
                    {
                        "name": "input_name",
                        "contentType": "text/plain",
                        "value": "text_value测试",
                    }
                ],
            }
        ]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a",
            b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a",
            b"input_name",
            b"\x00\x00\x00\x0a",
            b"text/plain",
            b"\x00\x00\x00\x0a",
            bytes("text_value测试", "utf-8"),
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF",  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"I"
        assert ret == expected

    def test_retrieve_msg_predict_binary(self, socket_patches):
        expected = [
            {
                "requestId": b"request_id",
                "headers": [],
                "parameters": [
                    {"name": "input_name", "contentType": "", "value": b"binary"}
                ],
            }
        ]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a",
            b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a",
            b"input_name",
            b"\x00\x00\x00\x00",
            b"\x00\x00\x00\x06",
            b"binary",
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF",  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"I"
        assert ret == expected

    def test_create_load_model_response(self):
        msg = codec.create_load_model_response(200, "model_loaded")

        assert msg == b"\x00\x00\x00\xc8\x00\x00\x00\x0cmodel_loaded\xff\xff\xff\xff"

    def test_create_predict_response(self):
        msg = codec.create_predict_response(["OK"], {0: "request_id"}, "success", 200)
        assert (
            msg
            == b"\x00\x00\x00\xc8\x00\x00\x00\x07success\x00\x00\x00\nrequest_id\x00\x00\x00\x00\x00\x00"
            b"\x00\xc8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02OK\xff\xff\xff\xff"
        )

    def test_create_predict_response_with_error(self):
        msg = codec.create_predict_response(None, {0: "request_id"}, "failed", 200)

        assert (
            msg
            == b"\x00\x00\x00\xc8\x00\x00\x00\x06failed\x00\x00\x00\nrequest_id\x00\x00\x00\x00\x00\x00\x00"
            b"\xc8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05error\xff\xff\xff\xff"
        )

    def test_create_predict_response_with_context(self):
        # context = MagicMock("Context")
        # context.stopping_criteria = {0: lambda x: True}
        # context.set_response_headers
        # get_response_headers
        from ts.context import Context, RequestProcessor

        ctx = Context(
            "model_name",
            "model_dir",
            "manifest",
            batch_size=2,
            gpu=0,
            mms_version=1.0,
        )
        ctx.stopping_criteria = {0: lambda _: True, 1: lambda _: False}
        ctx.request_processor = {0: RequestProcessor({}), 1: RequestProcessor({})}

        msg = codec.create_predict_response(
            ["OK", "NOT OK"],
            {0: "request_0", 1: "request_1"},
            "success",
            200,
            context=ctx,
        )

        def read_int(m):
            a = struct.unpack("!i", m[:4])[0]
            del msg[:4]
            return a

        def read_string(m, n):
            a = m[:n].decode("utf-8")
            del msg[:n]
            return a

        def read_map(m, n):
            ret = {}
            while n:
                l = read_int(m)
                k = read_string(m, l)
                l = read_int(m)
                v = read_string(m, l)
                ret[k] = v
                n -= 1
            return ret

        assert read_int(msg) == 200  # code

        assert read_int(msg) == 7  # msg length

        assert read_string(msg, 7) == "success"  # msg

        length = read_int(msg)
        expected = ["request_0", "false", "OK", "request_1", "true", "NOT OK"]
        while length != -1:
            req_id = read_string(msg, length)
            assert req_id == expected.pop(0)

            length = read_int(msg)
            content_type = read_string(msg, length)
            assert content_type == ""

            http_code = read_int(msg)
            assert http_code == 200

            length = read_int(msg)
            http_phrase = read_string(msg, length)
            assert http_phrase == ""

            length = read_int(msg)
            kv = read_map(msg, length)
            assert kv["ts_stream_next"] == expected.pop(0)

            length = read_int(msg)
            pred = read_string(msg, length)
            assert pred == expected.pop(0)

            length = read_int(msg)
        assert length == -1
