# FRI Benchmark

This repository contains the code from the paper "Towards Standardized Evaluation of Feasible Region Identification in Constrained Engineering Design".

## Supplementary Material
`supplementary_material.pdf` contains the the problem descriptions, experiment setup and additional figures as a supplement to the main paper.

## Setup UV
Install following the steps on https://docs.astral.sh/uv/getting-started/installation/ depending on your operating system.

Now you have uv on your machine. You can check it by running `uv` in terminal (after restarting it).

Clone the repository and `cd` into the folder.

Run `uv venv`. This will create a `.venv` folder in the repository with the virtual environment.
After that, run `uv sync` to install the dependencies as defined in `pyproject.toml`. `uv.lock` keeps track of the dependencies and installations.

## How To Use

Before running any script, make sure you activate venv environment and call the correct python.

This repository contains benchmark problems which are implemented in `problems.py` and can be used directly.

The implementation of the FR-IGD metric is given in `metrics/igd.py`. In order to calculate it, a reference set is needed. The reference sets used in the paper for the benchmark problems are provided in the `test_sets` folder as `reference_clusters_{problem_name}_{num_feas}_{num_feas_filtered}_{num_total}.pt` files. They are created by running `create_cluster_reference_set.py`.

Alongside the FR-IGD, `metrics/model_metrics.py` calcualtes all kinds of model performance metrics like MCC (also used in the paper). It requires the test sets `test_sets/test_set_{problem_name}_{num_feas}_{num_total}.pt`. These files are created using `create_test_sets.py`.

The acquisition functions that were used in the paper are implemnted in `acquisitions`.

An example usage of all parts is given in `example_runner.py`.

## Cite Us
```
@inproceedings{nikova2026towards,
title={Towards standardized evaluation of feasible region identification in constrained engineering design},
authors={Nikova, Ioana and Dhebar, Yashesh and {Rojas Gonzalez}, Sebastian and Dhaene, Tom and Couckuyt, Ivo},
book title = {To appear in IEEE Congress on Evolutionary Computation},
year={2026},
}
```
