# CC_OPF_Constraint_Informed_Randomness

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Repository Structure and Usage](#repository-structure-and-usage)
- [Citation](#citation)

## Overview

This repository contains all the implementations needed to reproduce the computational studies in the paper, which include simulating different datasets as randomness, optimization models to solve chance-constrained optimal power flow (cc-opf), out-of-sample reliabitilies validation. In addition, this repository contains the IEEE test cases system in MATPOWER format, where users can use their own preferred test cases or modify.

## Installation

Clone and install requirements for Python scripts:

```bash
git clone https://github.com/Subramanyam-Lab/Constraint_Informed_CCOPF
cd Constraint_Informed_CCOPF
conda create -n constraint_informed_ccopf python=3.9 -y
conda activate constraint_informed_ccopf
pip install -r requirements.txt
```

In the previously cloned folder, do the following to set up Julia environment:
```bash
git clone https://github.com/Subramanyam-Lab/Constraint_Informed_CCOPF
cd Constraint_Informed_CCOPF
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To install an academic license of the Gurobi solver, please visit (https://www.gurobi.com/academia/academic-program-and-licenses/).

## Repository Structure and Usage

### 1. Datasets

The datasets used for statistical fitting and OPF optimization model are both synthetic and real-world data.
- Gaussian synthetic: Synthetic forecast errors that are Gaussian distributed.  
- Cauchy synthetic: Synthetic forecast errors that are Cauchy distributed.
- NordPool Real: Sourced from NordPool production & production forecast over 2 weeks (15 consecutive days) in March. The file contains wind power generation forecasts and actual (in MW) data for 10 European regions at a 15-minute time window. In experiments, these errors will be normalized. 

### 2. Code (Estimation)
- Run this code to get a simple demo of the advantage of constraint-informed over classical on a synthetic Cauchy distributed data and to obtain results in Table 2 and Figure 1 from the paper.
```bash
python toy_example_methodology.py 
```

- Run these three scripts individually to generate plots for Figure 3 and obtain statistics for Table 3 from the paper.
```bash
python Gaussian_data_simulation.py
```
```bash
python Cauchy_data_simulation.py
```
```bash
python NordPoolSimu_ver2.py
```

- Run these scripts to get the PTDF matrix (in Julia) and use it to construct samples for 2D randomness eta_l for GMM estimation (in Python)
```bash
include("get_ptdf.jl")
```
```bash
python eta_l_estimate.py
```

- Run these three scripts to generate statistics in the Appendix.C by constraining zeromean for GMM estimation, as well as getting estimated parameters for 2D randomness eta_l
```bash
python Cauchy_data_simulation_FixedZeroMean.py
```
```bash
python NordPoolSimu_zeromean.py
```
```bash
python eta_l_estimate_FixedZeroMean.py
```

### 3. Code (Optimization)
Code for reformulated DC-OPF problem: 
- Run this script to implement the algorithm in [Fathabad et al (2023)](https://www.sciencedirect.com/science/article/pii/S0377221722004957) that finds the optimal placement of piecewise linear approximation by specifying a tolerance.
```bash
python pwl.py
```  
- Run this script to solve an OPF model and obtain worst-case constraint-violation:
```bash
include("constraint_violation.jl")
``` 


## Citation
If you have ever used the datasets and estimation/optimization codes, please cite the following paper: 
```bibtex
@article{yi2025chance,
  title={Chance-Constrained DC Optimal Power Flow Using Constraint-Informed Statistical Estimation},
  author={Yi, Tianyang and Maldonado, D Adrian and Subramanyam, Anirudh},
  journal={arXiv preprint arXiv:2508.21687},
  year={2025}
}
