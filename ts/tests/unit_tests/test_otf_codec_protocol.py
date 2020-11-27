# coding=utf-8




"""
On The Fly Codec tester
"""

from collections import namedtuple

import pytest

import ts.protocol.otf_message_handler as codec
from builtins import bytes


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket'])
    mock_patch = Patches(mocker.patch('socket.socket'))
    mock_patch.socket.recv.return_value = b'1'
    return mock_patch


# noinspection PyClassHasNoInit
class TestOtfCodecHandler:

    def test_retrieve_msg_unknown(self, socket_patches):
        socket_patches.socket.recv.side_effect = [b"U", b"\x00\x00\x00\x03"]
        with pytest.raises(ValueError, match=r"Invalid command: .*"):
            codec.retrieve_msg(socket_patches.socket)

    def test_retrieve_msg_load_gpu(self, socket_patches):
        expected = {"modelName": b"model_name", "modelPath": b"model_path",
                    "batchSize": 1, "handler": b"handler", "gpu": 1,
                    "envelope": b"envelope"}

        socket_patches.socket.recv.side_effect = [
            b"L",
            b"\x00\x00\x00\x0a", b"model_name",
            b"\x00\x00\x00\x0a", b"model_path",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x07", b"handler",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x08", b"envelope"
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"L"
        assert ret == expected

    def test_retrieve_msg_load_no_gpu(self, socket_patches):
        expected = {"modelName": b"model_name", "modelPath": b"model_path",
                    "batchSize": 1, "handler": b"handler",
                    "envelope": b"envelope"}

        socket_patches.socket.recv.side_effect = [
            b"L",
            b"\x00\x00\x00\x0a", b"model_name",
            b"\x00\x00\x00\x0a", b"model_path",
            b"\x00\x00\x00\x01",
            b"\x00\x00\x00\x07", b"handler",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x08", b"envelope"
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b"L"
        assert ret == expected

    def test_retrieve_msg_predict(self, socket_patches):
        expected = [{
            "requestId": b"request_id", "headers": [], "parameters": [
                {"name": "input_name",
                 "contentType": "application/json",
                 "value": {"data": "value"}
                 }
            ]
        }]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a", b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a", b"input_name",
            b"\x00\x00\x00\x0F", b"application/json",
            b"\x00\x00\x00\x0F", b'{"data":"value"}',
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF"  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b'I'
        assert ret == expected

    def test_retrieve_msg_predict_text(self, socket_patches):
        expected = [{
            "requestId": b"request_id", "headers": [], "parameters": [
                {"name": "input_name",
                 "contentType": "text/plain",
                 "value": u"text_value测试"
                 }
            ]
        }]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a", b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a", b"input_name",
            b"\x00\x00\x00\x0a", b"text/plain",
            b"\x00\x00\x00\x0a", bytes(u"text_value测试", "utf-8"),
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF"  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b'I'
        assert ret == expected

    def test_retrieve_msg_predict_binary(self, socket_patches):
        expected = [{
            "requestId": b"request_id", "headers": [], "parameters": [
                {"name": "input_name",
                 "contentType": "",
                 "value": b"binary"
                 }
            ]
        }]

        socket_patches.socket.recv.side_effect = [
            b"I",
            b"\x00\x00\x00\x0a", b"request_id",
            b"\xFF\xFF\xFF\xFF",
            b"\x00\x00\x00\x0a", b"input_name",
            b"\x00\x00\x00\x00",
            b"\x00\x00\x00\x06", b"binary",
            b"\xFF\xFF\xFF\xFF",  # end of parameters
            b"\xFF\xFF\xFF\xFF"  # end of batch
        ]
        cmd, ret = codec.retrieve_msg(socket_patches.socket)

        assert cmd == b'I'
        assert ret == expected

    def test_create_load_model_response(self):
        msg = codec.create_load_model_response(200, "model_loaded")

        assert msg == b'\x00\x00\x00\xc8\x00\x00\x00\x0cmodel_loaded\xff\xff\xff\xff'

    def test_create_predict_response(self):
        msg = codec.create_predict_response(["OK"], {0: "request_id"}, "success", 200)
        assert msg == b'\x00\x00\x00\xc8\x00\x00\x00\x07success\x00\x00\x00\nrequest_id\x00\x00\x00\x00\x00\x00' \
                      b'\x00\xc8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02OK\xff\xff\xff\xff'

    def test_create_predict_response_with_error(self):
        msg = codec.create_predict_response(None, {0: "request_id"}, "failed", 200)

        assert msg == b'\x00\x00\x00\xc8\x00\x00\x00\x06failed\x00\x00\x00\nrequest_id\x00\x00\x00\x00\x00\x00\x00' \
                      b'\xc8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05error\xff\xff\xff\xff'
