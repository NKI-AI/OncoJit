#!/usr/bin/env python
# coding=utf-8
"""The setup script."""
import ast

from setuptools import find_packages, setup  # type: ignore  # noqa

with open("oncojit/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


with open("README.rst") as readme_file:
    long_description = readme_file.read()

install_requires = ["torch>=2.0.1"]


setup(
    author="Ajey Pai Karkala",
    long_description=long_description,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "oncojit=oncojit.cli:main",
        ],
    },
    description="Accelerating Oncology Models with JIT Compilation",
    install_requires=install_requires,
    license="MIT License",
    include_package_data=True,
    name="oncojit",
    test_suite="tests",
    url="https://github.com/NKI-AI/OncoJit",
    py_modules=["oncojit"],
    version=version,
)
