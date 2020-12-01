# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]

setup(
    name='image_transformer',
    version='0.1.0',
    author_email='dsun20@bloomberg.net',
    license='../../LICENSE.txt',
    url='https://github.com/kubeflow/kfserving/docs/sameples/transformer',
    description='Transformer',
    long_description=open('README.md').read(),
    python_requires='>=3.6',
    packages=find_packages("image_transformer"),
    install_requires=[
        "kfserving>=0.2.1",
        "argparse>=1.4.0",
        "requests>=2.22.0",
        "joblib>=0.13.2",
        "pandas>=0.24.2",
        "numpy>=1.16.3",
        "kubernetes >= 9.0.0",
        "torchvision>=0.4.0",
        "pillow==7.1.0"
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)
