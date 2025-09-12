# CC_OPF_Constraint_Informed_Randomness

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Repository Structure and Usage](#repository-structure-and-usage)
- [Citation](#citation)

## Overview

This repository contains all the implementations needed to reproduce the computational studies in the paper, which include simulating different datasets as randomness, optimization models to solve chance-constrained optimal power flow (cc-opf), out-of-sample reliabitilies validation. In addition, this repository contains the IEEE test cases system in MATPOWER format, where users can use their own preferred test cases or modify.

## Installation

1. **Create and Activate Conda Environment**
```bash
conda create --name constraint_informed_ccopf python=3.9
conda activate constraint_informed_ccopf
```

2. **Install Packages(for example, scikit-learn)**
```bash
pip install scikit-learn
```

3. **Install Packages in Julia(for example, PowerModels)**
In a new Julia REPL, type the right square bracket ] to enter package mode
```bash
activate constraint_informed_ccopf
```
In the same package mode, add necessary packages:
```bash
add PowerModels
```

## Requirements

- Python 3.12.6+
- Julia v1.10.4+
- JuMP v1.22.1+
- Gurobi Solver v11.0.2+

### Pre-installed Package Requirement
In addition to basic Python and Julia Packages (e.g., `numpy`, `matplotlib`, `LinearAlgebra`, `pandas`), this repository relies heavily on the following packages:
- Python `scikit-learn`, in particular `sklearn.mixture` for GMM estimations
- Julia `PowerModels`, for parsing test cases and constructing a power system network

## Repository Structure and Usage

### 1. Datasets

The datasets used for statistical fitting and OPF optimization model are both synthetic and real-world data.
- Gaussian synthetic: Synthetic forecast errors that are Gaussian distributed.  
- Cauchy synthetic: Synthetic forecast errors that are Cauchy distributed.
- NordPool Real: Sourced from NordPool production & production forecast over 2 weeks (15 consecutive days) in March. The file contains wind power generation forecasts and actual (in MW) data for 10 European regions at a 15-minute time window. In experiemtns, these errors will be normalized. 

### 2. Code (Estimation)
Code for comparing estimation performances using classical and constraint-informed approaches:
- `code/estimation/toy_example_methodology.py`: run this code to get a simple demo of the advantage of constraint-informed over classical on a synthetic Cauchy distributed data. Results are in Table 2 and Figure 1.
```bash
python code/estimation/toy_example_methodology.py 
``` 
- `code/estimation/Gaussian_data_simulation.py`,`code/estimation/Cauchy_data_simulation.py`, `code/estimation/NordPoolSimu_ver2.py`: Run these codes to generate plots for Figure 3 and obtain statistics for Table 3.
```bash
python code/estimation/Gaussian_data_simulation.py
```
```bash
python code/estimation/Cauchy_data_simulation.py
```
```bash
python code/estimation/NordPoolSimu_ver2.py
``` 
- `code/estimation/get_ptdf.jl` and `code/estimation/eta_l_estimate.py`: are the files for getting PTDF matrix, and use that to construct 2D data samples for eta_l and further estimating a GMM.
- `code/estimation/Cauchy_data_simulation_FixedZeroMean.py`, `code/estimation/NordPoolSimu_zeromean.py`, `code/estimation/eta_l_estimate_FixedZeroMean.py`: Use these files to apply a modified EM algorithm to fix the means to be zero for the estimation.

### 3. Code (Optimization)
Code for reformulated DC-OPF problem: 
- `code/optimization/pwl.py`: Find the optimal placement of piecewise linear approximation by specificing a tolerance.
- `code/optimization/gaussian_reformulation.jl`: Reformulation for CC-OPF when using a single Gaussian distribution.
- `code/optimization/gmm_reformulation.jl`: Reformulation for CC-OPF when using GMM distribution.
- `code/optimization/constraint_violation.jl`: run OPF model and obtain worst-case constraint-violation.


## Citation
If you have ever used the datasets and estimation/optimization codes, please cite the following paper: 
```bibtex
@article{yi2025chance,
  title={Chance-Constrained DC Optimal Power Flow Using Constraint-Informed Statistical Estimation},
  author={Yi, Tianyang and Maldonado, D Adrian and Subramanyam, Anirudh},
  journal={arXiv preprint arXiv:2508.21687},
  year={2025}
}
