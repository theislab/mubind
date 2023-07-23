# mubind

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/main/docs/logo.png"
    width="400px" alt="mubind logo">
    </a>
</p>

<p align="center">
    <img src="docs/cartoon.png"
    width="400px" alt="mubind logo">
    </a>
</p>

## Model highlights

- Mubind is a machine learning method for learning motif associations with single cell genomics data, using graph representations such as k-nearest neighbors graph. The main codebase is based on PyTorch.
it allows learning binding modes (filters) and sample-sample relationships (graphs) that communicate filter activities across cells.
- This package works with single-cell genomics data, scATAC-seq, scChIP-seq, etc. We have also tested it on bulk in vitro samples (HT-SELEX, PBM). Please see the documentation for related examples.

## Worklflow

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/main/docs/workflow.png"
    width="750" alt="mubind workflow">
    </a>
</p>

## Model architecture

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/main/docs/architecture.png"
    width="550px" alt="mubind architecture">
    </a>
</p>


## Scalability

- The scalability of this method has been tested on single-cell datasets between 10,000 and 100,000 cells, with running times below 2 hours.
    
## Resources

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
pip install git+https://github.com/theislab/mubind.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please open an [Issue](https://github.com/theislab/mubind/issues).

## Citation

If mubind is useful for your research, please consider citing as:
```bibtex
@software{mubind,
author = {Ibarra, Schneeberger, Erdogan, Martens, Aliee, Klein and Theis FJ},
doi = {},
month = {},
title = {{mubind}},
url = {https://github.com/theislab/mubind},
year = {2023}
}
```

## Preprint

t.b.c.

[issue-tracker]: https://github.com/theislab/mubind/issues
[changelog]: https://mubind.readthedocs.io/latest/changelog.html
[link-docs]: https://mubind.readthedocs.io
[link-api]: https://mubind.readthedocs.io/latest/api.html

# Acknowledgments.

- [Wellcome Leap | Delta Tissue](https://wellcomeleap.org/delta-tissue/)
- [Helmholtz Zentrum Muenchen](https://www.helmholtz-munich.de/en/computational-health-center).


Project template created using [scverse cookie template](https://github.com/scverse/cookiecutter-scverse)
