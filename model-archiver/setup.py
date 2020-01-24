

# To build and upload a new version, follow the steps below.
# Notes:
# - this is a "Universal Wheels" package that is pure Python and supports both Python2 and Python3
# - Twine is a secure PyPi upload package
# - Make sure you have bumped the version! at ts/version.py
# $ pip install twine
# $ pip install wheel
# $ python setup.py bdist_wheel --universal

# *** TEST YOUR PACKAGE WITH TEST PI ******
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# If this is successful then push it to actual pypi

# $ twine upload dist/*

"""
Setup.py for the model-archiver tool
"""

from datetime import date
import sys
from setuptools import setup, find_packages
# pylint: disable = relative-import
import model_archiver

pkgs = find_packages()


def pypi_description():
    """Imports the long description for the project page"""
    with open('PyPiDescription.rst') as df:
        return df.read()


def detect_model_archiver_version():
    if "--release" in sys.argv:
        sys.argv.remove("--release")
        # pylint: disable = relative-import
        return model_archiver.__version__.strip()

    # pylint: disable = relative-import
    return model_archiver.__version__.strip() + 'b' + str(date.today()).replace('-', '')


if __name__ == '__main__':
    version = detect_model_archiver_version()
    requirements = ['future', 'enum-compat']

    setup(
        name='torch-model-archiver',
        version=version,
        description='Torch Model Archiver is used for creating archives of trained neural net models '
                    'that can be consumed by TorchServe inference',
        long_description=pypi_description(),
        author='PyTorch Serving team',
        author_email='noreply@noreply.com',
        url='https://github.com/pytorch/serve/model-archiver/',
        keywords='TorchServe Torch Model Archive Archiver Server Serving Deep Learning Inference AI',
        packages=pkgs,
        install_requires=requirements,
        entry_points={
            'console_scripts': ['torch-model-archiver=model_archiver.model_packaging:generate_model_archive']
        },
        include_package_data=True,
        license='Apache License Version 2.0'
    )
