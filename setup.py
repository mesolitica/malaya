import setuptools


__packagename__ = 'malaya'

setuptools.setup(
    name = __packagename__,
    packages = setuptools.find_packages(),
    version = '2.6.1',
    python_requires = '>=3.6.*',
    description = 'Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning Tensorflow.',
    author = 'huseinzol05',
    author_email = 'husein.zol05@gmail.com',
    url = 'https://github.com/huseinzol05/Malaya',
    download_url = 'https://github.com/huseinzol05/Malaya/archive/master.zip',
    keywords = ['nlp', 'bm'],
    install_requires = [
        'xgboost==0.80',
        'sklearn',
        'sklearn_crfsuite',
        'scikit-learn',
        'requests',
        'fuzzywuzzy',
        'tqdm',
        'unidecode',
        'tensorflow',
        'numpy',
        'scipy',
        'python-levenshtein',
        'PySastrawi',
        'toolz',
        'ftfy',
        'networkx',
    ],
    license = 'MIT',
    classifiers = [
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Text Processing',
    ],
    package_data = {
        'malaya': [
            '_utils/web/*.html',
            '_utils/web/static/*.js',
            '_utils/web/static/*.css',
        ]
    },
)
