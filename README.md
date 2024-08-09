# mubind

<p align="center">
    <img src="https://raw.githubusercontent.com/theislab/mubind/development/docs/_static/logo.png"
    width="400px" alt="mubind logo">
    </a>
</p>

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/development/docs/_static/concept_figure_repo.png?raw=true"
    width="400px" alt="mubind logo">
    </a>
</p>

## Model highlights

- MuBind is a deep learning model that can learn DNA-sequence features predictive of cell transitions in single-cell genomics data, using graph representations and sequence-activity across cells. The codebase is written in PyTorch.
- This package works with single-cell genomics data, scATAC-seq, etc. We have also tested it on bulk in vitro samples (HT-SELEX). See documentation for examples.
- Complemented with velocity-driven graph representations we learn sequence-to-activity transcriptional regulators linked with developmental processes. These predictions are biologically confirmed in several systems, and reinforced through chromatin accessibility and orthogonal gene expression data across pseudotemporal order. Refer to [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.07.605876v1) for more details.


## Workflow

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/development/docs/_static/workflow.png?raw=true"
    width="750" alt="mubind workflow">
    </a>
</p>

## Model architecture

<p align="center">
    <img src="https://github.com/theislab/mubind/blob/development/docs/_static/architecture.png?raw=true"
    width="550px" alt="mubind architecture">
    </a>
</p>


## Scalability

- Number of cells: The scalability of this method has been tested on single-cell datasets between 10,000 and 100,000 cells.
- Number of peaks: We have tested randomly selected features, or EpiScanpy's [variability score](https://episcanpy.readthedocs.io/en/anna/api/episcanpy.pp.select_var_feature.html). Modeling all features requires calibration of batch sizes and total GPU memory.
- Usual running times (one GPU): We get variable running times based on hyper-parameters, and they range between 10 minutes (prior filters) and 3 hours (de novo).
    
## Resources

Please refer to the [documentation](https://mubind.readthedocs.io/).

- [Tutorials](https://mubind.readthedocs.io/en/latest/tutorials.html)

## Installation

There are several alternative options to install mubind:

### pip

1. Install the latest release of `mubind` from `PyPI <https://pypi.org/project/mubind/>`_:

```bash
pip install mubind
```

2. Install the latest development version:

```bash
pip install git+https://github.com/theislab/mubind.git@main
```

### conda

Available soon.

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
