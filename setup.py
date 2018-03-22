#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='MSSA_Learning',
    version='0.0.0',
    license="GNU General Public License 3",
    description="MSSA_Learning",
    long_description=open('README.md').read(),
    install_requires= ["pandas", "numpy", "fastdtw", "progressbar"],
    package_dir={'': "src"},
    packages=["mssa_learning", "mssa_learning.tools", "mssa_learning.models"]
)
