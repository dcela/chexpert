# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='chexpredict',
    version='0.1.0',
    description='Multi-label classification of chest X-rays',
    long_description=readme,
    author='Devin Cela',
    author_email='dcela@ucsd.edu',
    url='https://github.com/dcela/chexpert',
    license=license,
    packages=find_packages(exclude=('docs'))
)
