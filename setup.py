from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ghgflux",
    version="0.0.1",
    author="Jamie McQuilkin",
    author_email="jamie.mcquilkin@postgrad.manchester.ac.uk",
    url="",
    license="",
    packages=find_packages(),
    description="Calculating gas fluxes from UAV data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.9',
)
