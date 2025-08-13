# CC_OPF_Constraint_Informed_Randomness

This repository corresponds to the following paper. 

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Repository Structure and Usage](#repository-structure-and-usage)

## Overview

This repository contains all the implementations needed to reproduce the computational studies in the paper, which include simulating different datasets as randomness, optimization models to solve chance-constrained optimal power flow (cc-opf), out-of-sample reliabitilies validation. In addition, this repository contains the IEEE test cases system in MATPOWER format, where users can use their own preferred test cases or modify.

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
- Gaussian synthetic: Originally used to validate classical and constraint-informed approaches when estimating using a Gaussian distribution. Since MLEs of Gaussian estimators are independent of the underlying distribution and are preserved after linear transformation, both methods will give the same results. Therefore, the parameters don't matter, and users can even change to other distributions (e.g., Weibull) if only the Gaussian distribution is used to fit the data points. 
- Cauchy synthetic: Synthetic forecast errors that are Cauchy distributed, of which the parameters are used to fit wind forecast errors of wind units in ERCOT.
- NordPool Real: Sourced from NordPool production & production forecast over 2 weeks (15 consecutive days) in March. The file contains wind power generation forecasts and actual (in MW) data for 10 European regions at a 15-minute time window. 

### 2. Code (Estimation)
Code for comparing estimation performances using classical and constraint-informed approaches:
- `code/estimation/toy_example_methodology.py`: Run this code to get a simple demo of the advantage of constraint-informed over classical on a synthetic Cauchy distributed data. Results are in Table 2 and Figure 1. 
- `code/estimation/Gaussian_data_simulation.py`,`code/estimation/Cauchy_data_simulation.py`, `code/estimation/NordPoolSimu_ver2.py`: Run these codes to generate plots for Figure 3 and obtain statistics for Table 3. 
- `code/estimation/get_ptdf.jl` and `code/estimation/eta_l_estimate.py`: are the files for getting PTDF matrix, and use that to construct 2D data samples for eta_l and further estimating a GMM.
- `code/estimation/Cauchy_data_simulation_FixedZeroMean.py`, `code/estimation/NordPoolSimu_zeromean.py`, `code/estimation/eta_l_estimate_FixedZeroMean.py`: Use these files to apply a modified EM algorithm to fix the means to be zero for the estimation.

### 3. Code (Optimization)
Code for reformulated DC-OPF problem: 
- `code/optimization/pwl.py`: Find the optimal placement of piecewise linear approximation by specificing a tolerance.
- `code/optimization/pwl.py`: 
