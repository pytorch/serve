

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
Setup.py for the workflow-archiver tool
"""

from datetime import date
import sys
from setuptools import setup, find_packages
# pylint: disable = relative-import
import workflow_archiver

pkgs = find_packages()


def pypi_description():
    """Imports the long description for the project page"""
    with open('PyPiDescription.rst') as df:
        return df.read()


def detect_workflow_archiver_version():
    if "--release" in sys.argv:
        sys.argv.remove("--release")
        # pylint: disable = relative-import
        return workflow_archiver.__version__.strip()

    # pylint: disable = relative-import
    return workflow_archiver.__version__.strip() + 'b' + str(date.today()).replace('-', '')


if __name__ == '__main__':
    version = detect_workflow_archiver_version()

    setup(
        name='torch-workflow-archiver',
        version=version,
        description='Torch Workflow Archiver is used for creating archives of workflow designed using'
                    ' trained neural net models that can be consumed by TorchServe inference',
        long_description=pypi_description(),
        author='PyTorch Serving team',
        author_email='noreply@noreply.com',
        url='https://github.com/pytorch/serve/workflow-archiver/',
        keywords='TorchServe Torch Workflow Archive Archiver Server Serving Deep Learning Inference AI',
        packages=pkgs,
        entry_points={
            'console_scripts': ['torch-workflow-archiver=workflow_archiver.workflow_packaging:generate_workflow_archive']
        },
        include_package_data=True,
        license='Apache License Version 2.0'
    )

