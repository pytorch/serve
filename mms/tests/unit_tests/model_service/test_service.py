# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import os
import shutil
import sys
import tempfile
import unittest
from io import BytesIO

import PIL
import mxnet as mx
import numpy as np
import pytest
from helper.pixel2pixel_service import UnetGenerator
from mms.model_service.mxnet_model_service import MXNetBaseService, GluonImperativeBaseService

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../..')


def empty_file(path):
    open(path, 'a').close()


def module_dir(tmpdir):
    path = '{}/test'.format(tmpdir)
    os.mkdir(path)
    empty_file('{}/test-symbol.json'.format(path))
    empty_file('{}/test-0000.params'.format(path))
    empty_file('{}/synset.txt'.format(path))

    with open('{}/signature.json'.format(path), 'w') as sig:
        signature = {
            "input_type": "image/jpeg",
            "inputs": [
                {
                    'data_name': 'data1',
                    'data_shape': [1, 3, 64, 64]
                },
                {
                    'data_name': 'data2',
                    'data_shape': [1, 3, 32, 32]
                }
            ],
            "output_type": "application/json",
            "outputs": [
                {
                    'data_name': 'softmax',
                    'data_shape': [1, 10]
                }
            ]
        }
        json.dump(signature, sig)

    return path


def create_symbolic_manifest(path):
    with open('{}/MANIFEST.json'.format(path), 'w') as man:
        manifest = {
            "Engine": {
                "MXNet": 0.12
            },
            "Model-Archive-Description": "test",
            "License": "Apache 2.0",
            "Model-Archive-Version": 0.1,
            "Model-Server": 0.1,
            "Model": {
                "Description": "test",
                "Service": "test",
                "Symbol": "",
                "Parameters": "test-0000.params",
                "Signature": "signature.json",
                "Model-Name": "test",
                "Model-Format": "MXNet-Symbolic"
            }
        }
        json.dump(manifest, man)


def create_imperative_manifest(path):
    with open('{}/MANIFEST.json'.format(path), 'w') as man:
        manifest = {
            "Engine": {
                "MXNet": 0.12
            },
            "Model-Archive-Description": "test",
            "License": "Apache 2.0",
            "Model-Archive-Version": 0.1,
            "Model-Server": 0.1,
            "Model": {
                "Description": "test",
                "Service": "test",
                "Symbol": "",
                "Parameters": "",
                "Signature": "signature.json",
                "Model-Name": "test",
                "Model-Format": "Gluon-Imperative"
            }
        }
        json.dump(manifest, man)


class TestService(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _train_and_export(self, path):
        model_path = curr_path + '/' + path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        num_class = 10
        data1 = mx.sym.Variable('data1')
        data2 = mx.sym.Variable('data2')
        conv1 = mx.sym.Convolution(data=data1, kernel=(2, 2), num_filter=2, stride=(2, 2))
        conv2 = mx.sym.Convolution(data=data2, kernel=(3, 3), num_filter=3, stride=(1, 1))
        pooling1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(1, 1), pool_type="avg")
        pooling2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(1, 1), pool_type="max")
        flatten1 = mx.sym.flatten(data=pooling1)
        flatten2 = mx.sym.flatten(data=pooling2)
        summary = mx.sym.sum(data=flatten1, axis=1) + mx.sym.sum(data=flatten2, axis=1)
        fc = mx.sym.FullyConnected(data=summary, num_hidden=num_class)
        sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')

        dshape1 = (10, 3, 64, 64)
        dshape2 = (10, 3, 32, 32)
        lshape = (10,)

        mod = mx.mod.Module(symbol=sym, data_names=('data1', 'data2'),
                            label_names=('softmax_label',))
        mod.bind(data_shapes=[('data1', dshape1), ('data2', dshape2)],
                 label_shapes=[('softmax_label', lshape)])
        mod.init_params()
        mod.init_optimizer(optimizer_params={'learning_rate': 0.01})

        data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                           mx.nd.random.uniform(5, 15, dshape2)],
                                     label=[mx.nd.ones(lshape)])
        mod.forward(data_batch)
        mod.backward()
        mod.update()
        with open('%s/synset.txt' % model_path, 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % i)

    def _write_image(self, img_arr):
        img_arr = mx.nd.transpose(img_arr, (1, 2, 0)).astype(np.uint8).asnumpy()
        mode = 'RGB'
        image = PIL.Image.fromarray(img_arr, mode)
        output = BytesIO()
        image.save(output, format='jpeg')
        return output.getvalue()

    def test_vision_init(self):
        path = 'test'
        self._train_and_export(path)
        model_path = curr_path + '/' + path
        os.system('rm -rf %s' % model_path)

    def test_vision_inference(self):
        path = 'test'
        self._train_and_export(path)

        os.system('rm -rf %s/test' % curr_path)

    def test_gluon_inference(self):
        path = 'gluon'
        model_name = 'gluon1'
        model_path = curr_path + '/' + path
        os.mkdir(model_path)
        ctx = mx.cpu()
        net_g = UnetGenerator(in_channels=3, num_downs=8)
        data = mx.nd.random_uniform(0, 255, shape=(1, 3, 256, 256))
        net_g.initialize(mx.init.Normal(0.02), ctx=ctx)
        net_g(data)
        net_g.save_params('%s/%s.params' % (model_path, model_name))
        with open('%s/signature.json' % model_path, 'w') as sig:
            signature = {
                "input_type": "image/jpeg",
                "inputs": [
                    {
                        'data_name': 'data',
                        'data_shape': [1, 3, 256, 256]
                    },
                ],
                "output_type": "image/jpeg",
                "outputs": [
                    {
                        'data_name': 'output',
                        'data_shape': [1, 3, 256, 256]
                    }
                ]
            }
            json.dump(signature, sig)

        cmd = 'python %s/../../export_model.py --model-name %s --model-path %s' \
              % (curr_path, model_name, model_path)
        os.system(cmd)

        os.system('rm -rf %s %s/%s.model %s/%s' % (model_path, os.getcwd(),
                                                   model_name, os.getcwd(), model_name))

    def test_mxnet_model_service(self):
        mod_dir = module_dir(self.test_dir)
        if mod_dir.startswith('~'):
            model_path = os.path.expanduser(mod_dir)
        else:
            model_path = mod_dir
        create_symbolic_manifest(model_path)
        manifest = json.load(open(os.path.join(model_path, 'MANIFEST.json')))
        with pytest.raises(Exception):
            MXNetBaseService('test', model_path, manifest)
        os.system('rm -rf %s' % model_path)

    def test_gluon_model_service(self):
        mod_dir = module_dir(self.test_dir)
        if mod_dir.startswith('~'):
            model_path = os.path.expanduser(mod_dir)
        else:
            model_path = mod_dir
        create_imperative_manifest(model_path)
        manifest = json.load(open(os.path.join(model_path, 'MANIFEST.json')))
        GluonImperativeBaseService('test', model_path, manifest,
                                   mx.gluon.model_zoo.vision.alexnet(pretrained=True))
        os.system('rm -rf %s' % model_path)

    def runTest(self):
        self.test_vision_init()
        self.test_vision_inference()
        self.test_gluon_inference()
        self.test_mxnet_model_service()
        self.test_gluon_model_service()
        self.test_incorrect_service()
