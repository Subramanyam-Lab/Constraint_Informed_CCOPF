# CC_OPF_Constraint_Informed_Randomness

This repository corresponds to the following paper. 

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

This repository contains all the implementations needed to reproduce the computational studies in the paper, which include simulating different datasets as randomness, optimization models to solve chance-constrained optimal power flow (cc-opf), out-of-sample reliabitilies validation. In addition, this repository contains the IEEE test cases system in MATPOWER format, where users can use their own preferred test cases or modify.

## Repository Structure

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


### 3. Code (Optimization)
Code for solving reformulate DC-OPF problem: 
- Includes pre-trained $\hat{\phi}$ and $\hat{\rho}$ models for GVS, PSCC, and RSCC sampling methods



1. **Clone the Repository**
```bash
git clone https://github.com/Subramanyam-Lab/NEO-LRP.git
cd neo-lrp
```

2. **Create and Activate Conda Environment**
```bash
conda create --name neo_lrp python=3.9
conda activate neo_lrp
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

To run the neural embedded framework:

```bash
python neo_lrp_execute.py \
    --BFS /path/to/solutions \
    --phi_loc /path/to/phi_model.onnx \
    --rho_loc /path/to/rho_model.onnx \
    --existing_excel_file results.xlsx \
    --sheet_name "Results" \
    --normalization dynamic
```

### Command Line Arguments

- `--BFS`: Directory path for storing integer feasible solutions (will not be used)
- `--phi_loc`: Path to the phi (φ) model file (ONNX format)
- `--rho_loc`: Path to the rho (ρ) model file (ONNX format)
- `--existing_excel_file`: Path to the Excel file for storing results
- `--sheet_name`: Name of the worksheet in the Excel file
- `--normalization`: Normalization strategy (`fixed` for GVS or `dynamic` for RSCC and PSCC)

## Results

The framework generates two Excel files in the `results/` directory:

### 1. `neo_lrp_results.xlsx`
Contains comprehensive solution metrics including:
- Instance details and solution costs (FLP, VRP, LRP)
- Number of routes in optimal solution
- Execution times (MIP+NN, initial solution, NN model)
- VRPSolverEasy computed costs and times
- Best known solutions (BKS)
- Optimization and prediction gaps

### 2. `flp.xlsx`
Contains FLP-specific metrics including:
- Instance details
- FLP and VRP costs
- Execution times for different components
- OR-Tools and VRPSolverEasy performance metrics
- Solution quality gaps


For questions and support, please open an issue in the repository or contact the authors.
