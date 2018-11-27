rm dist/*
python3 setup.py sdist bdist_wheel
twine upload dist/*

rm dist/*
python3 setup-gpu.py sdist bdist_wheel
twine upload dist/*
