# Inverse Problems in Proton Beam Imaging at ESS

This repository contains code developed in connection with the master’s
thesis *Inverse Problems in Proton Beam Imaging at ESS: Analysis and Numerical Methods* by Joel Henriksson, Lund University (LTH).

The code includes a simulator of the forward model that produces the proton beam images,
as well as two methods for parameter estimation: optimization and Markov chain
Monte Carlo (with parallel tempering). All code is based on the JAX library, developed by Google for high-performance computing.

## Reproducibility

The repository is version-controlled. The code corresponding to the submitted thesis has the
tag `v1.0-thesis`.

## Code structure
The code is organized into modules for configuration, simulation, noise modeling,
optimization, and statistical methods.


## Requirements
Python dependencies are listed in `requirements.txt`. The code is intended to be run on a GPU due to heavy parallelization.

## Scope
The repository primarily serves as an archive of the code used in the thesis.



