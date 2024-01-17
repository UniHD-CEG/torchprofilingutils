#!/usr/bin/env python3
#
# Copyright (C) 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
# Modifications copyright (C) 2023 Computing Systems Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils import util
from typing import Any

from setuptools import find_packages, setup  # type: ignore[import]

pkg_name = 'torch_profiling_utils'
author = 'Kevin Franz Stehle'
author_email = 'kevin.stehle@ziti.uni-heidelberg.de'
description = ('Torch Profiling Utils')
license = 'Apache 2.0'
install_requires: list[str] = ['numpy>=1',
                                'pandas>=1.4',
                                'pydot>=1.4',
                                'torchinfo>=1.8',
                                'fvcore>=0.1',
                                'bigtree>=0.14']

##############################################

setup(name=pkg_name,
        version='0.0.1',
        python_requires='>=3.9',
        include_package_data=True,
        setup_requires=[],
        install_requires=install_requires,
        author=author,
        author_email=author_email,
        description=description,
        license=license,)
