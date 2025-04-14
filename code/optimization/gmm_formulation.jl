using PowerModels
using JuMP
using Ipopt
using CSV
using DataFrames

network_data = PowerModels.parse_file("/Users/tianyangyi/Desktop/Constraint_Informed_CCOPF/data/case118.m")
all_gens = keys(network_data["gen"])

# List of 10 generator indices (from gen) to be replaced by wind units with no costs
renewable_gens = ["5", "11", "12", "25", "28", "29", "30", "37", "45", "51"]
for g in renewable_gens
    network_data["gen"][g]["cost"] = [0.0, 0.0, 0.0]
end

# List of condensers, subset of all gens
condenser_gens = [g for g in keys(network_data["gen"]) if network_data["gen"][g]["pg"] == 0]
conventional_gens = sort(setdiff(collect(all_gens), renewable_gens, condenser_gens))

for g in conventional_gens
    print(network_data["gen"][g]["cost"])
end

# === Extract cost coefficients for conventional generators === #
c0 = Dict{String, Float64}()
c1 = Dict{String, Float64}()
c2 = Dict{String, Float64}()

for g in conventional_gens
    cost = network_data["gen"][g]["cost"]
    c0[g] = cost[3]
    c1[g] = cost[2]
    c2[g] = cost[1]
end

#= println("\n=== Cost coefficients for conventional generators ===")
for g in conventional_gens
    println("Gen ", g, ": c2 = ", c2[g], ", c1 = ", c1[g], ", c0 = ", c0[g])
end =#

# Read the PWL segments file
pwl_df = CSV.read("pwl_segments.csv", DataFrame)
a_segments = collect(pwl_df.slope)
b_segments = collect(pwl_df.intercept)
piecewise_params = (a_segments, b_segments)

# Optional: print to verify
println("Loaded PWL segments:")
for (i, (a, b)) in enumerate(zip(a_segments, b_segments))
    println("Segment $i: a = $a, b = $b")
end

fmax = network_data["branch"][l]["rateA"]


model = Model(Ipopt.Optimizer)

# Decision variables
@variable(model, p̄[g in conventional_gens] >= 0)      # Nominal power output
@variable(model, α[g in conventional_gens] >= 0)       # Participation factors

# Objective
@objective(model, Min, sum(c2[g]*p̄[g]^2 + c1[g]*p̄[g] + c0[g] for g in conventional_gens))

# Constraints
@constraint(model, sum(α[g] for g in conventional_gens) == 1)
wind_forecast = Dict(g => network_data["gen"][g]["pg"] for g in renewable_gens)
total_wind = sum(wind_forecast[g] for g in renewable_gens)
total_demand = sum(network_data["load"][b]["pd"] for b in keys(network_data["load"]))
@constraint(model, sum(p̄[g] for g in conventional_gens) + total_wind == total_demand)
GMM_params = [(0.0, 0.1, 0.5), (0.1, 0.2, 0.5)]
K = length(GMM_params)
S = length(a_segments)
ϵ = 0.05

# Generator capacity chance constraints (Eqns 35–38)
@variable(model, M1[g in conventional_gens, k=1:K, s=1:S])
@variable(model, M2[g in conventional_gens, k=1:K, s=1:S])

for g in conventional_gens
    pmax = network_data["gen"][g]["pmax"]
    pmin = network_data["gen"][g]["pmin"]
    for k in 1:K
        (μk, σk, πk) = GMM_params[k]
        for s in 1:S
            a_s, b_s = a_segments[s], b_segments[s]
            # Upper limit approximation
            @constraint(model, M1[g, k, s] <= a_s * (pmax - p̄[g] + μk * α[g]) + b_s * σk * α[g])
            # Lower limit approximation
            @constraint(model, M2[g, k, s] <= a_s * (p̄[g] - μk * α[g] - pmin) + b_s * σk * α[g])
        end
        # Piecewise minimum over all s
        @constraint(model, sum(GMM_params[k][3] * minimum([M1[g, k, s] for s in 1:S])) >= (1 - ϵ) * σk * α[g])
        @constraint(model, sum(GMM_params[k][3] * minimum([M2[g, k, s] for s in 1:S])) >= (1 - ϵ) * σk * α[g])
    end
end

# Line flow constraints (Eqns 47–50)
branches = keys(network_data["branch"])
@variable(model, M3[l in branches, k=1:K, s=1:S])
@variable(model, M4[l in branches, k=1:K, s=1:S])

PTDF = PowerModels.calc_basic_ptdf_matrix(network_data)

for l in branches
    Hl = PTDF[l, :]
    fmax = network_data["branch"][l]["rateA"]
    for k in 1:K
        (μk, σk, πk) = GMM_params[k]
        for s in 1:S
            a_s, b_s = a_segments[s], b_segments[s]
            μflow = sum(Hl[g] * p̄[g] for g in conventional_gens) - μk * sum(Hl[g] * α[g] for g in conventional_gens)
            σflow = σk * sqrt(sum((Hl[g] * α[g])^2 for g in conventional_gens))
            @constraint(model, M3[l, k, s] <= a_s * (fmax - μflow) + b_s * σflow)
            @constraint(model, M4[l, k, s] <= a_s * (μflow - fmax) + b_s * σflow)
        end
        @constraint(model, sum(GMM_params[k][3] * minimum([M3[l, k, s] for s in 1:S])) >= (1 - ϵ) * σk)
        @constraint(model, sum(GMM_params[k][3] * minimum([M4[l, k, s] for s in 1:S])) >= (1 - ϵ) * σk)
    end
end

