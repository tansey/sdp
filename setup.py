"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tf-sdp',
    version='0.9',
    description='Deep Nonparametric Estimation of Discrete Conditional Distributions via Smoothed Dyadic Partitioning',
    long_description=long_description,
    url='https://github.com/tansey/sdp',
    author='Wesley Tansey',
    author_email='wes.tansey@gmail.com',
    license='LGPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='statistics machinelearning tensorflow deeplearning smoothing',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'scipy', 'tensorflow'],
    package_data={
        'tfsdp': [],
    },
    entry_points={
    },
    ext_modules=[]
)













