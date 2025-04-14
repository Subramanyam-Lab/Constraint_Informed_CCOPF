using PowerModels
using JuMP
using Ipopt
using LinearAlgebra

ϵg = 0.05
ϵf = 0.05

function parse_case(file_name)
    data = make_basic_network(parse_file(file_name))
    return data
end

function compute_ptdf_matrix(data)
    ptdf_matrix = PowerModels.calc_basic_ptdf_matrix(data)
    return ptdf_matrix
end

function extract_gen_data(data)
    gen_data = data["gen"]
    num_gens = length(gen_data)

    Pgmin = [gen_data[string(i)]["pmin"] for i in 1:num_gens]
    Pgmax = [gen_data[string(i)]["pmax"] for i in 1:num_gens]
    gen_cost_a = [gen_data[string(i)]["cost"][1] for i in 1:num_gens]  
    gen_cost_b = [gen_data[string(i)]["cost"][2] for i in 1:num_gens]  
    gen_cost_c = [gen_data[string(i)]["cost"][3] for i in 1:num_gens]  

    return Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c
end

function extract_load_data(data)
    bus_data = data["bus"]
    load_data = data["load"]
    num_buses = length(bus_data)

    Pd = zeros(num_buses)
    for (i, load) in load_data
        Pd[load["load_bus"]] += load["pd"]
    end

    return Pd
end

function create_gen_ptdf_matrix(ptdf_matrix, gen_data)
    gen_to_bus = [gen_data[string(i)]["gen_bus"] for i in 1:length(gen_data)]
    gen_ptdf_matrix = ptdf_matrix[:, gen_to_bus]
    return gen_ptdf_matrix
end

function add_power_balance_constraint(model, Pg, Pd)
    @constraint(model, sum(Pg) == sum(Pd))
end

function add_participation_factor_constraint(model, α)
    @constraint(model, sum(α) == 1)
end

# Equations 7-10
function add_generation_chance_constraints(model, Pg, α, Pgmin, Pgmax, as, bs, wk, mu_k, sigma_k, πj, µj, ϵg, K, J, S)
    num_gens = length(Pg)
    
    @variable(model, M1[i=1:num_gens, k=1:K, s=1:S] >= 0)
    @variable(model, M2[i=1:num_gens, k=1:K, s=1:S] >= 0)
    @variable(model, β[i=1:num_gens, j=1:J], Bin)
    
    for i in 1:num_gens
        for k in 1:K
            for s in 1:S
                @constraint(model, M1[i, k, s] >= as[s] * (Pgmax[i] - (Pg[i] - α[i] * mu_k[k])) + bs[s] * (α[i] * sigma_k[k]))
                @constraint(model, M2[i, k, s] >= as[s] * ((Pg[i] - α[i] * mu_k[k]) - Pgmin[i]) + bs[s] * (α[i] * sigma_k[k]))
            end
        end
        for j in 1:J
            @constraint(model, β[i, j] --> {Pgmin[i] <= Pg[i] - α[i] * µj[j] <= Pgmax[i]})
        end
        @constraint(model, sum(wk[k] * (M1[i, k, s] + M2[i, k, s]) for k in 1:K) + sum(πj[j] * β[i, j] for j in 1:J) >= 2 - ϵg)
    end
end


# Equations 11-14
function add_line_flow_chance_constraints(model, gen_ptdf_matrix, Pg, as, bs, wk, mu_k, sigma_k, πj, µj, ϵf, K, J, S, branch_data)
    num_branches = size(gen_ptdf_matrix, 1)
    
    @variable(model, M3[l=1:num_branches, k=1:K, s=1:S] >= 0)
    @variable(model, M4[l=1:num_branches, k=1:K, s=1:S] >= 0)
    @variable(model, ζ[l=1:num_branches, j=1:J], Bin)
    
    for l in 1:num_branches
        for k in 1:K
            for s in 1:S
                @constraint(model, M3[l, k, s] >= as[s] * (branch_data[string(l)]["rate_a"] - h1[k, s]) + bs[s] * sqrt(h1[l] * sigma_k[k] * h1[l]))
                @constraint(model, M4[l, k, s] >= as[s] * (h1[k, s] - branch_data[string(l)]["rate_a"]) + bs[s] * sqrt(h1[l] * sigma_k[k] * h1[l]))
            end
        end
        for j in 1:J
            @constraint(model, ζ[l, j] --> {fmin[l] <= µj[j] <= fmax[l]})
        end
        @constraint(model, sum(wk[k] * (M3[l, k, s] + M4[l, k, s]) for k in 1:K) + sum(πj[j] * ζ[l, j] for j in 1:J) >= 2 - ϵf)
    end
end

function solve_cc_opf(Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c, Pd, gen_ptdf_matrix, branch_data)
    num_gens = length(Pgmin)
    num_branches = size(gen_ptdf_matrix, 1)


    # Dummy DGMM
    K = 3  
    J = 2  
    wk = [0.3, 0.4, 0.3]
    mu_k = [0.1, 0.2, 0.3]
    sigma_k = [0.05, 0.05, 0.05]
    πj = [0.5, 0.5]
    µj = [0.1, -0.1]

    S = 5
    as = [0.37624480013911055, 0.2653057449341264, 0.1318989182501275, 0.04622130525249513, 0.011412734704756934]
    bs = [0.5, 0.5665634331229905, 0.7266516251437891, 0.8808713285395274, 0.9644118978540991]

    model = Model(Ipopt.Optimizer)

    @variable(model, Pgmin[i] <= Pg[i=1:num_gens] <= Pgmax[i])
    @variable(model, α[i=1:num_gens] >= 0)

    @objective(model, Min, sum(gen_cost_a[i] * Pg[i]^2 + gen_cost_b[i] * Pg[i] + gen_cost_c[i] for i in 1:num_gens))

    add_power_balance_constraint(model, Pg, Pd)
    add_participation_factor_constraint(model, α)
    add_generation_chance_constraints(model, Pg, α, Pgmin, Pgmax, as, bs, wk, mu_k, sigma_k, πj, µj, ϵg, K, J, S)
    add_line_flow_chance_constraints(model, gen_ptdf_matrix, Pg, as, bs, wk, mu_k, sigma_k, πj, µj, ϵf, K, J, S, branch_data)

    optimize!(model)

    optimal_generation = value.(Pg)
    optimal_participation_factors = value.(α)
    return optimal_generation, optimal_participation_factors
end

function main(file_name)
    data = parse_case(file_name)
    ptdf_matrix = compute_ptdf_matrix(data)
    #println("PTDF Matrix:")
    #println(ptdf_matrix)

    Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c = extract_gen_data(data)
    Pd = extract_load_data(data)
    branch_data = data["branch"]

    gen_ptdf_matrix = create_gen_ptdf_matrix(ptdf_matrix, data["gen"])

    optimal_generation, optimal_participation_factors = solve_cc_opf(Pgmin, Pgmax, gen_cost_a, gen_cost_b, gen_cost_c, Pd, gen_ptdf_matrix, branch_data)
    println("Optimal Generator Outputs: ", optimal_generation)
    println("Optimal Participation Factors: ", optimal_participation_factors)
end

file_name = "/Users/tianyangyi/Desktop/OPFmixture/data/case9.m"
main(file_name)