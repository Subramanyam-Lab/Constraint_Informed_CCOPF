using PowerModels, JuMP, Gurobi, CSV, DataFrames

function replace_with_wind!(data::Dict, renewable_gens::Vector{String})
    for g in renewable_gens
        data["gen"][g]["cost"] = [0.0, 0.0, 0.0]
    end
    return data
end

function extract_controllable_generators(data::Dict, renewable_gens::Vector{String})
    all_gens = collect(keys(data["gen"]))
    condenser_gens = [g for g in all_gens if data["gen"][g]["pg"] == 0.0]
    controllable = sort(setdiff(all_gens, union(renewable_gens, condenser_gens)))
    return controllable
end

function extract_cost_coefficients(data::Dict, conventional_gens::Vector{String})
    c2 = Dict{String, Float64}()
    c1 = Dict{String, Float64}()
    c0 = Dict{String, Float64}()
    for g in conventional_gens
        cost = data["gen"][g]["cost"]
        c2[g] = cost[1]
        c1[g] = cost[2]
        c0[g] = cost[3]
    end
    return c0, c1, c2
end

function objective_cost!(model, pbar, alpha, c1, c2, beta_Ω, m_Ω, σ_Ω)
    μΩ = sum(beta_Ω[k] * m_Ω[k] for k in 1:length(beta_Ω))
    VarΩ = sum(beta_Ω[k] * (σ_Ω[k]^2 + m_Ω[k]^2) for k in 1:length(beta_Ω)) - μΩ^2
    @objective(model, Min,
        sum(
            c2[g] * (pbar[g] - alpha[g] * μΩ)^2 +
            c2[g] * alpha[g]^2 * VarΩ +
            c1[g] * (pbar[g] - alpha[g] * μΩ)
        for g in keys(pbar))
    )
end

function add_power_balance!(model, pbar, wind, demand)
    @constraint(model,
        sum(pbar[g] for g in keys(pbar)) +
        sum(values(wind)) ==
        sum(demand[b] for b in keys(demand))
    )
end

function add_gen_reformulations!(model, alpha, pbar, pmin, pmax,
                                 beta_Ω, m_Ω, σ_Ω, a, b, ε)
    K = length(beta_Ω)
    S = length(a)
    @variable(model, M1[g in keys(pbar), k in 1:K] >= 0)
    @variable(model, M2[g in keys(pbar), k in 1:K] >= 0)
    for g in keys(pbar)
        @constraint(model, sum(beta_Ω[k] * M1[g,k] for k in 1:K) ≥ 1 - ε)
        @constraint(model, sum(beta_Ω[k] * M2[g,k] for k in 1:K) ≥ 1 - ε)
        for k in 1:K, s in 1:S
            @constraint(model,
                M1[g,k,s] <= a[s] * (pmax[g] - pbar[g] + m_Ω[k] * alpha[g]) +
                           b[s] * σ_Ω[k] * alpha[g]
            )
            @constraint(model,
                M2[g,k,s] <= a[s] * (pbar[g] - m_Ω[k] * alpha[g] - pmin[g]) +
                           b[s] * σ_Ω[k] * alpha[g]
            )
        end
        @constraint(model,
            sum(beta_Ω[k] * minimum([M1[g,k,s] for s in 1:S]) for k in 1:K) >=
            (1 - ε) * sum(beta_Ω[k] * σ_Ω[k] * alpha[g] for k in 1:K)
        )
        @constraint(model,
            sum(beta_Ω[k] * minimum([M2[g,k,s] for s in 1:S]) for k in 1:K) >=
            (1 - ε) * sum(beta_Ω[k] * σ_Ω[k] * alpha[g] for k in 1:K)
        )
    end
end
