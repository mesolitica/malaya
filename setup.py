import setuptools


__packagename__ = 'malaya'


def readme():
    with open('README-pypi.rst') as f:
        return f.read()


with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='4.9.2.1',
    python_requires='>=3.6.*',
    description='Natural-Language-Toolkit for bahasa Malaysia, powered by Tensorflow and PyTorch.',
    long_description=readme(),
    author='huseinzol05',
    author_email='husein.zol05@gmail.com',
    url='https://github.com/huseinzol05/Malaya',
    download_url='https://github.com/huseinzol05/Malaya/archive/master.zip',
    keywords=['nlp', 'bm'],
    install_requires=req,
    license='MIT',
    classifiers=[
            'Programming Language :: Python :: 3.6',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Topic :: Text Processing',
    ],
    package_data={
        'malaya': [
            'function/web/*.html',
            'function/web/static/*.js',
            'function/web/static/*.css',
        ]},
)
