#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = [
    "pytest>=3",
]

setup(
    author="Mathis Hallier, Davide Gurnari",
    author_email="mathis.hallier28@gmail.com",
    python_requires=">=3.6",
    description="ClusterGraph is a tool which allows the visualization of a geometric organization of clusters.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="clustergraph",
    name="clustergraph",
    packages=find_packages(include=["clustergraph", "clustergraph.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dioscuri-tda/clustergraph",
    version="0.2",
    zip_safe=False,
)
