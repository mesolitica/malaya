import os
from distutils.core import setup
from distutils import dir_util
from distutils import sysconfig
import setuptools


__packagename__ = 'malaya'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.6.0.3',
    description='Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning.',
    author='huseinzol05',
    author_email='husein.zol05@gmail.com',
    url='https://github.com/DevconX/Malaya',
    download_url='https://github.com/DevconX/Malaya/archive/master.zip',
    keywords=['nlp', 'bm'],
    install_requires=[
          'xgboost==0.80',
          'sklearn',
          'scikit-learn==0.19.1',
          'requests',
          'fuzzywuzzy',
          'tqdm',
          'nltk',
          'unidecode',
          'numpy',
          'scipy',
          'python-levenshtein',
          'pandas',
          'tensorflow==1.5'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Text Processing',
    ],
    long_description=open('readme-pypi.srt').read())
