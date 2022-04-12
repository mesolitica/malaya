# Unit tests

<p align="left">
<a href="#"><img alt="coverage" src="coverage.svg"></a>
</p>

## how-to

1. Install pytest,

```bash
pip3 install pytest pytest-cov
```

2. Run pytest,

```bash
pytest tests --cov --cov-report term --cov-report html
```

Or run for specific py file,

```bash
pytest tests/test_emotion.py --cov --cov-report term --cov-report html
```

Or run for specific function,

```bash
pytest tests/test_emotion.py::test_multinomial --cov --cov-report term --cov-report html
```