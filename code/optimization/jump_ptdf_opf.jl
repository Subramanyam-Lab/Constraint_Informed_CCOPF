### This is the deterministic DC_OPF solution obtained by doing the following
# 1. use PowerModels to parse data into basic network format (i.e., the matrix format)
# 2. use PowerModels to compute PTDF matrix from this pre-processed data
# 3. use  JuMP to code up a deterministic DC-OPF using PTDF matrix, this formulation 
#       does not require phase angles as variaables
# 4. solve and compare power generation against groundtruth model

using PowerModels
using JuMP
using Ipopt
using LinearAlgebra

# Function to parse the MATPOWER case file
function parse_case(file_name)
    # Need the basic network format for matrix formulations
    data = make_basic_network(parse_file(file_name))
    return data
end

# Function to compute the PTDF matrix from basic dictionary
function compute_ptdf_matrix(data)
    ptdf_matrix = PowerModels.calc_basic_ptdf_matrix(data)
    return ptdf_matrix
end

# Function to get generator data
function get_gen_data(data)
    gen_data = data["gen"]
    num_gens = length(gen_data)

    # Lower and upper bound for generation variables
    Pgmin = [gen_data[string(i)]["pmin"] for i in 1:num_gens]
    Pgmax = [gen_data[string(i)]["pmax"] for i in 1:num_gens]

    # Quadratic, linear, and constant term for objective function
    gen_cost_a = [gen_data[string(i)]["cost"][1] for i in 1:num_gens]  
    gen_cost_b = [gen_data[string(i)]["cost"][2] for i in 1:num_gens]  
    gen_cost_c = [gen_data[string(i)]["cost"][3] for i in 1:num_gens]  

    return Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c
end

# Function to get load data
function get_load_data(data)
    bus_data = data["bus"]
    load_data = data["load"]
    num_buses = length(bus_data)

    Pd = zeros(num_buses)
    for (i, load) in load_data
        Pd[load["load_bus"]] += load["pd"]
    end

    # Need Pd for power balance
    return Pd
end

# Function to create PTDF matrix for generators, simplify matrix multiplication in line flow limit constraints
function create_gen_ptdf_matrix(ptdf_matrix, gen_data)
    gen_to_bus = [gen_data[string(i)]["gen_bus"] for i in 1:length(gen_data)]
    gen_ptdf_matrix = ptdf_matrix[:, gen_to_bus]
    return gen_ptdf_matrix
end

# Function to solve deterministic DC-OPF with the PTDF formulation
function solve_ptdf_dc_opf(Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c, Pd, gen_ptdf_matrix, branch_data)
    num_gens = length(Pgmin)
    num_branches = size(gen_ptdf_matrix, 1)

    model = Model(Ipopt.Optimizer)
    @variable(model, Pgmin[i] <= Pg[i=1:num_gens] <= Pgmax[i])
    @objective(model, Min, sum(gen_cost_a[i] * Pg[i]^2 + gen_cost_b[i] * Pg[i] + gen_cost_c[i] for i in 1:num_gens))

    # Power balance 
    @constraint(model, sum(Pg) == sum(Pd))

    # Line flow limits using PTDF matrix
    for l in 1:num_branches
        rate_a = branch_data[string(l)]["rate_a"]
        # Constraint Hp <= f 
        @constraint(model, -rate_a <= sum(gen_ptdf_matrix[l, i] * Pg[i] for i in 1:num_gens) <= rate_a)
    end

    optimize!(model)
    optimal_generation = value.(Pg)
    return optimal_generation
end

function main(file_name)
    data = parse_case(file_name)
    ptdf_matrix = compute_ptdf_matrix(data)
    Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c = get_gen_data(data)
    Pd = get_load_data(data)
    branch_data = data["branch"]
    gen_ptdf_matrix = create_gen_ptdf_matrix(ptdf_matrix, data["gen"])
    optimal_generation = solve_ptdf_dc_opf(Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c, Pd, gen_ptdf_matrix, branch_data)
    println("Optimal Generator Outputs: ", optimal_generation)
end

file_name = "data/case9.m"
main(file_name)