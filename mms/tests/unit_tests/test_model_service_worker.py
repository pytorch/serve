# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ModelServiceWorker is the worker that is started by the MMS front-end.
"""

import socket
from collections import namedtuple

import mock
import pytest
from mock import Mock

from mms.model_service_worker import MXNetModelServiceWorker
from mms.service import Service


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket'])
    mock_patch = Patches(mocker.patch('socket.socket'))
    mock_patch.socket.recv.side_effect = [
        b"L",
        b"\x00\x00\x00\x0a", b"model_name",
        b"\x00\x00\x00\x0a", b"model_path",
        b"\x00\x00\x00\x01",
        b"\x00\x00\x00\x07", b"handler",
        b"\x00\x00\x00\x01"
    ]
    return mock_patch


@pytest.fixture()
def model_service_worker(socket_patches):
    model_service_worker = MXNetModelServiceWorker('unix', 'my-socket', None, None)
    model_service_worker.sock = socket_patches.socket
    model_service_worker.service = Service('name', 'mpath', 'testmanifest', None, 0, 1)
    return model_service_worker


# noinspection PyClassHasNoInit
class TestInit:
    socket_name = "sampleSocketName"

    def test_missing_socket_name(self):
        with pytest.raises(ValueError, match="Incomplete data provided.*"):
            MXNetModelServiceWorker()

    def test_socket_in_use(self, mocker):
        remove = mocker.patch('os.remove')
        path_exists = mocker.patch('os.path.exists')
        remove.side_effect = OSError()
        path_exists.return_value = True

        with pytest.raises(Exception, match=r".*socket already in use: sampleSocketName.*"):
            MXNetModelServiceWorker('unix', self.socket_name)

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['remove', 'socket'])
        patches = Patches(
            mocker.patch('os.remove'),
            mocker.patch('socket.socket')
        )
        return patches

    def test_success(self, patches):
        MXNetModelServiceWorker('unix', self.socket_name)
        patches.remove.assert_called_once_with(self.socket_name)
        patches.socket.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)


# noinspection PyClassHasNoInit
class TestRunServer:
    accept_result = (mock.MagicMock(), None)

    def test_with_socket_bind_error(self, socket_patches, model_service_worker):
        bind_exception = socket.error("binding error")
        socket_patches.socket.bind.side_effect = bind_exception
        with pytest.raises(Exception):
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_not_called()

    def test_with_timeout(self, socket_patches, model_service_worker):
        exception = socket.timeout("Some Exception")
        socket_patches.socket.accept.side_effect = exception

        with pytest.raises(socket.timeout):
            model_service_worker.run_server()
        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()

    def test_with_run_server_debug(self, socket_patches, model_service_worker, mocker):
        exception = Exception("Some Exception")
        socket_patches.socket.accept.side_effect = exception
        mocker.patch('mms.model_service_worker.DEBUG', True)
        model_service_worker.handle_connection = Mock()

        with pytest.raises(Exception):
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()

    def test_success(self, model_service_worker):
        model_service_worker.sock.accept.return_value = self.accept_result
        model_service_worker.sock.recv.return_value = b""
        with pytest.raises(SystemExit):
            model_service_worker.run_server()
        model_service_worker.sock.accept.assert_called_once()


# noinspection PyClassHasNoInit
class TestLoadModel:
    data = {'modelPath': b'mpath', 'modelName': b'name', 'handler': b'handled'}

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['loader'])
        patches = Patches(mocker.patch('mms.model_service_worker.ModelLoaderFactory'))
        return patches

    def test_load_model(self, patches, model_service_worker):
        patches.loader.get_model_loader.return_value = Mock()
        model_service_worker.load_model(self.data)
        patches.loader.get_model_loader.assert_called()

    # noinspection PyUnusedLocal
    @pytest.mark.parametrize('batch_size', [(None, None), ('1', 1)])
    @pytest.mark.parametrize('gpu', [(None, None), ('2', 2)])
    def test_optional_args(self, patches, model_service_worker, batch_size, gpu):
        data = self.data.copy()
        if batch_size[0]:
            data['batchSize'] = batch_size[0]
        if gpu[0]:
            data['gpu'] = gpu[0]
            model_service_worker.load_model(data)


# noinspection PyClassHasNoInit
class TestHandleConnection:
    data = {'modelPath': b'mpath', 'modelName': b'name', 'handler': b'handled'}

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple("Patches", ["retrieve_msg"])
        patches = Patches(
            mocker.patch("mms.model_service_worker.retrieve_msg")
        )
        return patches

    def test_handle_connection(self, patches, model_service_worker):
        patches.retrieve_msg.side_effect = [(b"L", ""), (b"I", ""), (b"U", "")]
        model_service_worker.load_model = Mock()
        service = Mock()
        service.context = None
        model_service_worker.load_model.return_value = (service, "", 200)
        cl_socket = Mock()

        with pytest.raises(ValueError, match=r"Received unknown command.*"):
            model_service_worker.handle_connection(cl_socket)

        cl_socket.send.assert_called()
