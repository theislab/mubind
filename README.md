# mubind

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/workflow/status/ilibarra/mubind/Test/main
[link-tests]: https://github.com/theislab/mubind/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/mubind

<p align="center">
    <img src="docs/logo.png"
    width="400px" alt="mubind logo">
    </a>
</p>

Modeling for fitting and interpretation of biomolecular binding data.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [Tutorials][link-tutorial] and
-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`\_.

There are several alternative options to install mubind:

1. Install the latest release of `mubind` from `PyPI <https://pypi.org/project/mubind/>`_:

```bash
pip install mubind
```

2. Install the latest development version:

```bash
pip install git+https://github.com/ilibarra/mubind.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If mubind is useful for your research, please consider to cite as:
```bibtex
@software{mubind,
author = {Ibarra, Schneeberger},
doi = {},
month = {},
title = {{mubind}},
url = {https://github.com/theislab/mubind},
year = {2022}
}
```

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/ilibarra/mubind/issues
[changelog]: https://mubind.readthedocs.io/latest/changelog.html
[link-docs]: https://mubind.readthedocs.io
[link-api]: https://mubind.readthedocs.io/latest/api.html

## Old text

# mubind

Inference of binding specificities from protein sequence and multiple genomics experiments.

### SELEX data for testing [5 examples]

-   [Dropbox](https://www.dropbox.com/s/yqj4rvs6z24qdh4/selex.tar.gz?dl=0)
-   add to `annotations/selex`

# Installation

1. `conda env create -f environment.yml`
2. `conda activate mubind`
3. `pip install -e .`

Notes

-   The usage of `cookiecutter` requires `git >=2.38`
    -   `conda install -c anaconda git`
