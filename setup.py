import os
from setuptools import setup, find_packages

def read_requirements(filename):
    with open(os.path.join('requirements', filename)) as fp:
        return fp.read().strip().splitlines()

setup(
    name="inverse-astar-search",
    version="0.0.1",
    description="Course Project for Graduate Artificial Intelligence (CMU 15780 Spring 2020)",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    test_requires=read_requirements("dev.txt"),
    install_requires=read_requirements("prod.txt"),
    python_requires=">=3.7",
)
