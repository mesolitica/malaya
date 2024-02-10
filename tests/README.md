# Unit Tests

[![codecov](https://codecov.io/gh/huseinzol05/malaya/branch/master/graph/badge.svg?token=Mto5hHr8da)](https://codecov.io/gh/huseinzol05/malaya)

This folder contains the unit tests for malaya. We use `pytest` as our testing framework.

## Preqrequisites
> Note: You should use Python 3.6 and above with pip3. Make sure to also use a virtual environment!
Install pytest, pytest-cov, pytest-codecov, and gitpython.

```bash
pip3 install pytest pytest-cov pytest-codecov gitpython
```

Make sure tensorflow is installed, if some tests fail you might need to install the extra libraries in requirements.txt.

```bash
pip3 install -r requirements.txt
```

## Running Tests

**Run all tests**

```bash
pytest tests --cov --cov-report term --cov-report html
```

**Run failed tests only**

```bash
pytest tests --cov --cov-report term --cov-report html --last-failed
```

**Run a specific test file**

```bash
pytest tests/test_name.py --cov --cov-report term --cov-report html
```

**Run a test for a specific function**

```bash
pytest tests/test_name.py::test_multinomial --cov --cov-report term --cov-report html
```

## Updating Code Coverage
Once you have run all the tests upload the generated HTML coverage report to our CodeCov at https://app.codecov.io/gh/huseinzol05/malaya.

```bash
CODECOV_TOKEN=
pytest tests --cov --codecov --codecov-token=$CODECOV_TOKEN
```