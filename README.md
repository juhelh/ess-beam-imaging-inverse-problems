# Inverse Problems in Proton Beam Imaging at ESS

This repository contains code developed in connection with the master’s
thesis *Inverse Problems in Proton Beam Imaging at ESS: Analysis and Numerical Methods* by Joel Henriksson, Lund University (LTH).

The code contains a simulator of the forward model that produces the proton beam images,
as well as two methods for parameter estimation: optimization and Markov Chain
Monte Carlo (with parallel tempering). All code is JAX-based and meant to be run on a
GPU to be fast and efficient.

## Reproducibility

The repository is version-controlled. The code corresponding to the submitted thesis has the
tag `v1.0-thesis`.

## Code structure
The code is organized into modules for configuration, simulation, noise modeling,
optimization, and statistical methods. Scripts used to generate numerical results
and figures appearing in the thesis are included.


## Requirements
Python dependencies are listed in `requirements.txt`. The code is intended to be
run in a scientific Python environment with JAX support.


## Scope
The repository is primarily meant to serve as an archive of the code used in the thesis. 
However it may develop further over time.


