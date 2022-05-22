# multibind

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/ilibarra/multibind/Test/main
[link-tests]: https://github.com/theislab/multibind/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/multibind

ML for biomolecular binding

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`\_.

There are several alternative options to install multibind:

<!--
1) Install the latest release of `multibind` from `PyPI <https://pypi.org/project/multibind/>`_:

```bash
pip install multibind
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/ilibarra/multibind.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/ilibarra/multibind/issues
[changelog]: https://multibind.readthedocs.io/latest/changelog.html
[link-docs]: https://multibind.readthedocs.io
[link-api]: https://multibind.readthedocs.io/latest/api.html

## Old text

# multibind

PyTorch learning of molecular binding modes using multiple genomics data sources

### SELEX data for testing [5 examples]

-   [Dropbox](https://www.dropbox.com/s/yqj4rvs6z24qdh4/selex.tar.gz?dl=0)
-   add to `annotations/selex`

# Installation

1. `conda env create -f environment.yml`
2. `conda activate multibind`
3. `pip install -e .`

Notes

-   The usage of `cookiecutter` requires `git >=2.38`
    -   `conda install -c anaconda git`
