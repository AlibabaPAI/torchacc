#!/usr/bin/env python
import os

from setuptools import find_packages, setup

# Parse version number from torchacc/__init__.py:
with open('torchacc/__init__.py') as f:
    info = {}
    for line in f:
        if line.startswith('version'):
            exec(line, info)
            break


setup_info = dict(
    name=os.environ.get('TORCH_ACC_PACKAGE_NAME', 'torchacc'),
    version=info['version'],
    author='Alibaba Inc.',
    url='http://gitlab.alibaba-inc.com/torchx/torchacc',
    description='Torch Accelerator powered by Alibaba',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    # Package info
    packages=['torchacc'] + ['torchacc.' + \
                             pkg for pkg in find_packages('torchacc')],

    # Add _ prefix to the names of temporary build dirs
    options={'build': {'build_base': '_build'}, },
    zip_safe=True,
)

setup(**setup_info)
