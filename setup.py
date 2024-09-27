# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
import site
import sys

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# To use a consistent encoding
from codecs import open  # noqa
from os import path  # noqa

from setuptools import find_packages, setup  # noqa

here = path.abspath(path.dirname(__file__))

# dependencies only required during test
test = [
    "pytest",
    "pytest-cov",
    "ci_sdr",
]

try:
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="tssep",
    version="0.0.0",
    description="TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MERL",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="audio speech",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "einops",
        "sacred",
        "psutil",
        "pyyaml>=5.1",
        "tensorboardX",
        "torch",
        "torchvision",
        "torchaudio",
        "lazy_dataset",
        "nara_wpe>=0.0.11",
        "paderbox",
        "padertorch",
        "pb_bss @ git+http://github.com/fgnt/pb_bss",
    ],
    extras_require={
        "all": test,
    },
)
