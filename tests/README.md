# Unit tests

<p align="left">
<a href="#"><img alt="coverage" src="coverage.svg"></a>
</p>

## how-to

1. Install pytest,

```bash
pip install pytest pytest-cov
```

2. Run pytest,

```bash
COLOR=green
COVERAGE=$(pytest . --cov --cov-report term --cov-report html | grep TOTAL | awk '{print $4+0}')
wget "https://img.shields.io/badge/coverage-${COVERAGE}%25-${COLOR}.svg" -O coverage.svg
```

