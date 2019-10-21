# Contributing

Contributions are welcome and are greatly appreciated! Every little bit helps, and credit will always be given.

**Table of Contents**

- [Types of Contributions](#types-of-contributions)
  - [Report Bugs](#report-bugs)
  - [Fix Bugs](#fix-bugs)
  - [Implement Features](#implement-features)
  - [Dataset](#dataset)
  - [Improve Documentation](#improve-documentation)
  - [Submit Feedback](#submit-feedback)
- [Documentation](#documentation)
- [Local development environment](#local-development-environment)
  - [Installation](#installation)
- [Pull Request Guidelines](#pull-request-guidelines)

# Types of Contributions

## Report Bugs

Report bugs through [Github issue](https://github.com/huseinzol05/Malaya/issues/new).

Please report relevant information and preferably code that exhibits the problem.

## Fix Bugs

Look through the [Github issue](https://github.com/huseinzol05/Malaya/issues/new) for bugs. Anything is open to whoever wants to implement it.

## Implement Features

Look through the [Github issue](https://github.com/huseinzol05/Malaya/issues/new) or [Malaya-project](https://github.com/huseinzol05/Malaya/projects/1) for features. Any unassigned `improvement` issue is open to whoever wants to implement it.

Remember, **100% Tensorflow (version 1.10 and above, not 2.0, yet), no Keras**.

## Dataset

Create a new issue in [Github issue](https://github.com/huseinzol05/Malaya/issues/new) related to your data including the data link or attached it there. If you want to improve current dataset we have, you can check at [Malaya-Dataset](https://github.com/huseinzol05/Malaya-Dataset).

Or, you can simply email your data if you do not want to expose the data to public. Malaya will not exposed your data, but we will exposed our trained models based on your data.

Thanks to,

1. [Fake news](https://github.com/huseinzol05/Malaya-Dataset#fake-news), contributed by [syazanihussin](https://github.com/syazanihussin/FLUX/tree/master/data)
2. [Speech voice](https://github.com/huseinzol05/Malaya-Dataset#tolong-sebut), contributed by [Khalil Nooh](https://www.linkedin.com/in/khalilnooh/)
3. [Speech voice](https://github.com/huseinzol05/Malaya-Dataset#tolong-sebut), contributed by [Mas Aisyah Ahmad](https://www.linkedin.com/in/mas-aisyah-ahmad-b46508a9/)
4. [Singlish text dump](https://github.com/huseinzol05/malaya-dataset#singlish-text), contributed by [brytjy](https://github.com/brytjy)
5. [Singapore news](https://github.com/huseinzol05/malaya-dataset#singapore-news), contributed by [brytjy](https://github.com/brytjy)

## Improve Documentation

Malaya could always use better documentation, might have some typos or uncorrect object names.

## Submit Feedback

The best way to send feedback is to open an issue on [Github issue](https://github.com/huseinzol05/Malaya/issues/new).

## Unit test

Test every possible program flow! You can check [unit tests here](https://github.com/huseinzol05/Malaya/tree/master/tests).

# Documentation

The latest API documentation is usually available [here](https://malaya.readthedocs.io/en/latest/index.html). To generate a local version,

```
pip install readthedocs
cd docs
bash generate_template.sh
```

#  Local development environment

When you develop Malaya you can create local `virtualenv` with all requirements required by Malaya.

Advantage of local installation is that everything works locally, you do not have to enter Docker/container environment and you can easily debug the code locally. You can also have access to python `virtualenv` that contains all the necessary requirements and use it in your local IDE - this aids autocompletion, and running tests directly from within the IDE.

The disadvantage is that you have to maintain your dependencies and local environment consistent with other development environments that you have on your local machine.

It's also very difficult to make sure that your local environment is consistent with other environments. This can often lead to "works for me" syndrome. It's better to use the Docker Compose integration test environment in case you want reproducible environment consistent with other people.

## Installation

Install Python (3.6 and above) by using system-level package managers like yum, apt-get for Linux, or Homebrew for Mac OS at first.

# Pull Request Guidelines

Before you submit a pull request from your forked repo, check that it meets these guidelines:

1. The pull request should include step-by-step if the request is `improvement`.
2. Please [rebase your fork](http://stackoverflow.com/a/7244456/1110993), squash commits, and resolve all conflicts.
3. The pull request should work for Python 3.6 and above.
4. As Malaya grows as a project, we try to enforce a more consistent style and try to follow the Python
community guidelines. We currently enforce to use [BlackMamba](https://github.com/mohtar/blackmamba) for code standard.
