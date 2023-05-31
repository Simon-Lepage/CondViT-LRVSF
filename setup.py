#!/usr/bin/env python3

import setuptools
import os
from typing import List


here = os.path.abspath(os.path.dirname(__file__))


def _read_reqs(relpath: str) -> List[str]:
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


setuptools.setup(
    name="lrvsf",
    version="0.0.1",
    install_requires=_read_reqs("requirements.txt"),
    packages=setuptools.find_packages(),
    zip_safe=False,
    include_package_data=True,
)
