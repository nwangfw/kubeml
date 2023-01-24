from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='serverlessdl',
    version='1.0.1',
    description='Python tools for training Neural Networks in a serverless setup',
    author='Ning Wang',
    author_email="nwang@futurewei.com",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=['serverlessdl'],
    install_requires=[
        'torch>=1.7',
        'redisai>=1.0.1',
        'pymongo>=3.11.1',
        'flask>=1.1.2'
    ],
    license="MIT",
)
