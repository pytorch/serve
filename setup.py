

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
Setup.py for the model server package
"""

import errno
import os
import subprocess
import sys
from datetime import date
from shutil import copy2, rmtree

import setuptools.command.build_py
from setuptools import setup, find_packages, Command

import ts

pkgs = find_packages()

def pypi_description():
    """
    Imports the long description for the project page
    """
    with open('PyPiDescription.rst') as df:
        return df.read()


def detect_model_server_version():
    sys.path.append(os.path.abspath("ts"))
    if "--release" in sys.argv:
        sys.argv.remove("--release")
        return ts.__version__.strip()

    return ts.__version__.strip() + 'b' + str(date.today()).replace('-', '')


class BuildFrontEnd(setuptools.command.build_py.build_py):
    """
    Class defined to run custom commands.
    """
    description = 'Build Model Server Frontend'
    source_server_file = os.path.abspath('frontend/server/build/libs/server-1.0.jar')
    dest_file_name = os.path.abspath('ts/frontend/model-server.jar')

    # noinspection PyMethodMayBeStatic
    def run(self):
        """
        Actual method called to run the build command
        :return:
        """
        front_end_bin_dir = os.path.abspath('.') + '/ts/frontend'
        try:
            os.mkdir(front_end_bin_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(front_end_bin_dir):
                pass
            else:
                raise

        if os.path.exists(self.source_server_file):
            os.remove(self.source_server_file)

        # Remove build/lib directory.
        if os.path.exists('build/lib/'):
            rmtree('build/lib/')

        try:
            subprocess.check_call('frontend/gradlew -p frontend clean build', shell=True)
        except OSError:
            assert 0, "build failed"
        copy2(self.source_server_file, self.dest_file_name)


class BuildPy(setuptools.command.build_py.build_py):
    """
    Class to invoke the custom command defined above.
    """

    def run(self):
        sys.stderr.flush()
        self.run_command('build_frontend')
        setuptools.command.build_py.build_py.run(self)


class BuildPlugins(Command):
    description = 'Build Model Server Plugins'
    user_options = [('plugins=', 'p', 'Plugins installed')]
    source_plugin_dir = \
        os.path.abspath('plugins/build/plugins')

    def initialize_options(self):
        self.plugins = None

    def finalize_options(self):
        if self.plugins is None:
            print("No plugin option provided. Defaulting to 'default'")
            self.plugins = "default"

    # noinspection PyMethodMayBeStatic
    def run(self):
        if os.path.isdir(self.source_plugin_dir):
            rmtree(self.source_plugin_dir)

        try:
            if self.plugins == "endpoints":
                subprocess.check_call('plugins/gradlew -p plugins clean bS', shell=True)
            else:
                raise OSError("No such rule exists")
        except OSError:
            assert 0, "build failed"

        self.run_command('build_py')


if __name__ == '__main__':
    version = detect_model_server_version()

    requirements = ['Pillow', 'psutil', 'future']

    setup(
        name='torchserve',
        version=version,
        description='TorchServe is a tool for serving neural net models for inference',
        author='PyTorch Serving team',
        author_email='noreply@noreply.com',
        long_description=pypi_description(),
        url='https://github.com/pytorch/serve.git',
        keywords='TorchServe PyTorch Serving Deep Learning Inference AI',
        packages=pkgs,
        cmdclass={
            'build_frontend': BuildFrontEnd,
            'build_plugins': BuildPlugins,
            'build_py': BuildPy,
        },
        install_requires=requirements,
        entry_points={
            'console_scripts': [
                'torchserve=ts.model_server:start',
                'torchserve-export=ts.export_model:main'
            ]
        },
        include_package_data=True,
        license='Apache License Version 2.0'
    )
