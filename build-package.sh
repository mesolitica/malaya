rm dist/*
python3 setup.py sdist bdist_wheel
twine upload dist/*
anaconda upload dist/malaya-*.tar.gz

rm dist/*
python3 setup-gpu.py sdist bdist_wheel
twine upload dist/*
anaconda upload dist/malaya-gpu-*.tar.gz
