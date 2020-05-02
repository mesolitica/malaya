import setuptools


__packagename__ = 'malaya-gpu'

setuptools.setup(
    name = __packagename__,
    packages = setuptools.find_packages(),
    version = '3.4.2',
    python_requires = '>=3.6.*',
    description = 'Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning Tensorflow. GPU Version',
    author = 'huseinzol05',
    author_email = 'husein.zol05@gmail.com',
    url = 'https://github.com/huseinzol05/Malaya',
    download_url = 'https://github.com/huseinzol05/Malaya/archive/master.zip',
    keywords = ['nlp', 'bm'],
    install_requires = [
        'dateparser',
        'sklearn',
        'scikit-learn',
        'requests',
        'unidecode',
        'tensorflow-gpu>=1.14,<2.0',
        'numpy',
        'scipy',
        'PySastrawi',
        'ftfy',
        'networkx',
        'sentencepiece',
        'bert-tensorflow',
        'tqdm',
        'herpetologist',
        'youtokentome',
        'albert-tensorflow',
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
            'function/web/*.html',
            'function/web/static/*.js',
            'function/web/static/*.css',
        ]
    },
)
