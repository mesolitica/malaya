# Unit tests

[![codecov](https://codecov.io/gh/huseinzol05/malaya/branch/master/graph/badge.svg?token=Mto5hHr8da)](https://codecov.io/gh/huseinzol05/malaya)

## how-to

1. Install pytest,

```bash
pip3 install pytest pytest-cov pytest-codecov gitpython
```

2. Run pytest,

```bash
pytest tests --cov --cov-report term --cov-report html
```

Or run failed tests only,

```bash
pytest tests --cov --cov-report term --cov-report html --last-failed
```

Or run for specific py file,

```bash
pytest tests/test_emotion.py --cov --cov-report term --cov-report html
```

Or run for specific function,

```bash
pytest tests/test_emotion.py::test_multinomial --cov --cov-report term --cov-report html
```

3. Upload to CodeCov, https://app.codecov.io/gh/huseinzol05/malaya

```
CODECOV_TOKEN=
pytest tests --cov --codecov --codecov-token=$CODECOV_TOKEN
```