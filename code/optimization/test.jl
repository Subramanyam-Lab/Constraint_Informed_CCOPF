using PowerModels
using JuMP
using Gurobi
using LinearAlgebra

function extract_controllable_generators(data::Dict, wind_idx::Vector{String})::Vector{String}
    all_gens = collect(keys(data["gen"]))
    controllable = [g for g in all_gens if !(g in wind_idx) && data["gen"][g]["pg"] > 0.0]
    return sort(controllable)
end

function extract_cost_coefficients(data::Dict, gens::Vector{String})
    c2 = Dict{String,Float64}()
    c1 = Dict{String,Float64}()
    for g in gens
        cost = data["gen"][g]["cost"] 
        c2[g] = cost[1]
        c1[g] = cost[2]
    end
    return c1, c2
end


function parse_network(case_file::String, wind_idx::Vector{String})
    data   = PowerModels.parse_file(case_file)
    buses  = collect(keys(data["bus"]))
    gens   = collect(keys(data["gen"]))
    lines  = collect(keys(data["branch"]))
    nb, ng, nl = length(buses), length(gens), length(lines)

    bus_idx  = Dict(buses[i] => i for i in 1:nb)
    gen_idx  = Dict(gens[i]  => i for i in 1:ng)
    line_idx = Dict(lines[i] => i for i in 1:nl)

    total_demand = sum(data["load"][ld]["pd"] for ld in keys(data["load"]))
    println("Total system demand: ", total_demand)

    p_gen0 = zeros(nb)
    for g in gens
        bi = bus_idx[string(data["gen"][g]["gen_bus"])]
        p_gen0[bi] += data["gen"][g]["pg"]
    end

    H = PowerModels.calc_basic_ptdf_matrix(data)
    println("PTDF size: ", size(H))

    hgen = [zeros(ng) for _ in 1:nl]
    for l in lines
        li = line_idx[l]
        for g in gens
            bi = bus_idx[string(data["gen"][g]["gen_bus"])]
            hgen[li][gen_idx[g]] = -H[li, bi]
        end
    end

    p_wind_bar = zeros(nb)
    for g in wind_idx
        bi = bus_idx[string(data["gen"][g]["gen_bus"])]
        p_wind_bar[bi] += data["gen"][g]["pg"]
    end
    println("Sample p_wind_bar[1:5]: ", p_wind_bar[1:min(end,5)])
    println("Sum of p_wind_bar (total wind injection): ", sum(p_wind_bar))

    f0 = H * p_gen0

    return (
        data=data,
        buses=buses, gens=gens, lines=lines,
        bus_idx=bus_idx, gen_idx=gen_idx, line_idx=line_idx,
        total_demand=total_demand,
        p_gen0=p_gen0, p_wind_bar=p_wind_bar,
        H=H, hgen=hgen, f0=f0
    )
end

function add_variables(model::Model, gens::Vector{String}, lines::Vector{String}, K::Int, S::Int)
    @variable(model, pBar[g in gens] >= 0)
    @variable(model, α[g in gens] >= 0)
    @constraint(model, sum(α[g] for g in gens) == 1)
    @variable(model, δ[l in lines] >= 0)
    @variable(model, M1[g in gens, k in 1:K, s in 1:S] >= 0)
    @variable(model, M2[g in gens, k in 1:K, s in 1:S] >= 0)
    @variable(model, M3[l in lines, k in 1:K, s in 1:S] >= 0)
    @variable(model, M4[l in lines, k in 1:K, s in 1:S] >= 0)
    return (pBar=pBar, α=α, δ=δ, M1=M1, M2=M2, M3=M3, M4=M4)
end

function add_objective(model::Model, gens::Vector{String}, vars, gmm_Ω, c1::Dict, c2::Dict)
    EΩ = sum(gmm_Ω[:β][k]*gmm_Ω[:m][k] for k in 1:length(gmm_Ω[:β]))
    VΩ = sum(gmm_Ω[:β][k]*(gmm_Ω[:σ][k]^2 + gmm_Ω[:m][k]^2)
             for k in 1:length(gmm_Ω[:β])) - EΩ^2
    @objective(model, Min,
        sum(c2[g]*(vars.pBar[g] - vars.α[g]*EΩ)^2 +
            c2[g]*vars.α[g]^2*VΩ +
            c1[g]*(vars.pBar[g] - vars.α[g]*EΩ)
            for g in gens))
end

function add_gen_constraints(model::Model, gens::Vector{String}, vars, gmm_Ω, pwl::Dict, ϵ::Float64)
    S = length(pwl[:a])
    for g in gens
        pmin = model[:_data]["gen"][g]["pmin"]
        pmax = model[:_data]["gen"][g]["pmax"]
        for k in 1:length(gmm_Ω[:β])
            μk, σk, βk = gmm_Ω[:m][k], gmm_Ω[:σ][k], gmm_Ω[:β][k]
            @constraint(model, vars.pBar[g] - μk*vars.α[g] <= pmax)
            exprU = (pmax - vars.pBar[g] + μk*vars.α[g]) / σk
            @constraint(model, sum(βk*vars.M1[g,k,s] for s in 1:S) >= (1-ϵ)*vars.α[g])
            @constraint(model, [s=1:S], vars.M1[g,k,s] <= pwl[:a][s]*exprU + pwl[:b][s]*vars.α[g])
            @constraint(model, vars.pBar[g] - μk*vars.α[g] >= pmin)
            exprL = (vars.pBar[g] - pmin - μk*vars.α[g]) / σk
            @constraint(model, sum(βk*vars.M2[g,k,s] for s in 1:S) >= (1-ϵ)*vars.α[g])
            @constraint(model, [s=1:S], vars.M2[g,k,s] <= pwl[:a][s]*exprL + pwl[:b][s]*vars.α[g])
        end
    end
end

function add_line_constraints(model::Model, cont_gens::Vector{String}, parsed, vars, gmm_η, pwl::Dict, ϵ::Float64)
    S = length(pwl[:a])
    for (li,l) in enumerate(parsed.lines)
        rate = parsed.data["branch"][l]["rate_a"]
        γ = sum(parsed.hgen[li][parsed.gen_idx[g]] * vars.α[g] for g in cont_gens)
        C0 = gmm_η[l][:C0]
        @constraint(model, vars.δ[l]^2 >= γ^2*C0[1,1] + 2*γ*C0[1,2] + C0[2,2])
        for k in 1:length(gmm_η[l][:λ])
            νk, τk, λk = gmm_η[l][:ν][:,k], gmm_η[l][:τ][k], gmm_η[l][:λ][k]
            @constraint(model, parsed.f0[li] + dot(νk,[γ,1]) <= rate)
            exprU = (rate - parsed.f0[li] - dot(νk,[γ,1])) / τk
            @constraint(model, sum(λk*vars.M3[l,k,s] for s in 1:S) >= (1-ϵ)*vars.δ[l])
            @constraint(model, [s=1:S], vars.M3[l,k,s] <= pwl[:a][s]*exprU + pwl[:b][s]*vars.δ[l])
            @constraint(model, parsed.f0[li] + dot(νk,[γ,1]) >= -rate)
            exprL = (rate + parsed.f0[li] + dot(νk,[γ,1])) / τk
            @constraint(model, sum(λk*vars.M4[l,k,s] for s in 1:S) >= (1-ϵ)*vars.δ[l])
            @constraint(model, [s=1:S], vars.M4[l,k,s] <= pwl[:a][s]*exprL + pwl[:b][s]*vars.δ[l])
        end
    end
end

function build_ccopf_constraint_informed(case_file::String,
                                         wind_idx::Vector{String},
                                         gmm_Ω::Dict,
                                         gmm_η::Dict{String,Dict},
                                         pwl::Dict,
                                         ϵ::Float64)
    parsed    = parse_network(case_file, wind_idx)
    cont_gens = extract_controllable_generators(parsed.data, wind_idx)
    c1, c2    = extract_cost_coefficients(parsed.data, cont_gens)

    model = Model(Gurobi.Optimizer)
    model[:_data] = parsed.data

    K, S = length(gmm_Ω[:β]), length(pwl[:a])
    vars    = add_variables(model, cont_gens, parsed.lines, K, S)

    @constraint(model,
        sum(vars.pBar[g] for g in cont_gens) +
        sum(parsed.p_wind_bar) == parsed.total_demand)

    add_objective(model, cont_gens, vars, gmm_Ω, c1, c2)
    add_gen_constraints(model,   cont_gens, vars, gmm_Ω, pwl, ϵ)
    add_line_constraints(model, cont_gens, parsed, vars, gmm_η, pwl, ϵ)

    return model
end
