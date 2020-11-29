import setuptools


__packagename__ = 'malaya'


def readme():
    with open('README-pypi.rst', 'rb') as f:
        return f.read().decode('UTF-8')


setuptools.setup(
    name = __packagename__,
    packages = setuptools.find_packages(),
    version = '4.0.4',
    python_requires = '>=3.6.*',
    description = 'Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning Tensorflow.',
    long_description = readme(),
    author = 'huseinzol05',
    author_email = 'husein.zol05@gmail.com',
    url = 'https://github.com/huseinzol05/Malaya',
    download_url = 'https://github.com/huseinzol05/Malaya/archive/master.zip',
    keywords = ['nlp', 'bm'],
    install_requires = [
        'dateparser',
        'scikit-learn',
        'requests',
        'unidecode',
        'tensorflow>=1.14,<2.0',
        'numpy',
        'scipy',
        'ftfy',
        'networkx',
        'sentencepiece',
        'bert-tensorflow',
        'tqdm',
        'herpetologist',
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
