from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="SIENA",
    version="0.1",
    author="Joana Godinho",
    description="Method to assess differential expression over single-cell data.",
    packages=find_packages(),
    install_requires=requirements,
)
