using PowerModels
using JuMP
using LinearAlgebra
using MathOptInterface

# Network construction
function extract_controllable_generators(data::Dict, wind_idx::Vector{String})::Vector{String}
    all_gens = collect(keys(data["gen"]))
    return sort([g for g in all_gens if !(g in wind_idx) && data["gen"][g]["pg"] > 0.0])
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
    data  = PowerModels.parse_file(case_file)
    buses = collect(keys(data["bus"]))
    gens  = collect(keys(data["gen"]))
    lines = collect(keys(data["branch"]))
    nb, ng, nl = length(buses), length(gens), length(lines)

    bus_idx  = Dict(buses[i] => i for i in 1:nb)
    gen_idx  = Dict(gens[i]  => i for i in 1:ng)
    line_idx = Dict(lines[i] => i for i in 1:nl)

    total_demand = sum(data["load"][l]["pd"] for l in keys(data["load"]))

    p_wind_bar = zeros(nb)
    for g in wind_idx
        bi = bus_idx[string(data["gen"][g]["gen_bus"])];
        p_wind_bar[bi] += data["gen"][g]["pg"]
    end

    load_bus = zeros(nb)
    for ld in keys(data["load"])
        bi = bus_idx[string(data["load"][ld]["load_bus"])];
        load_bus[bi] += data["load"][ld]["pd"]
    end

    net_base = p_wind_bar .- load_bus

    # PTDF
    H = PowerModels.calc_basic_ptdf_matrix(data)

    # constant nominal flow under net_base
    f0_const = H * net_base

    # line‐gen participation mapping
    hgen = [zeros(ng) for _ in 1:nl]
    for l in lines
        li = line_idx[l]
        for g in gens
            bi = bus_idx[string(data["gen"][g]["gen_bus"])];
            hgen[li][gen_idx[g]] = -H[li, bi]
        end
    end

    return (
        data=data,
        buses=buses, gens=gens, lines=lines,
        bus_idx=bus_idx, gen_idx=gen_idx, line_idx=line_idx,
        total_demand=total_demand,
        p_wind_bar=p_wind_bar,
        H=H, hgen=hgen, f0_const=f0_const
    )
end

# Optimization Model
function add_variables(model::Model, gens::Vector{String}, lines::Vector{String}, K::Int, S::Int)
    @variable(model, 0 <= pBar[g in gens] <= model[:_data]["gen"][g]["pmax"])
    @variable(model, α[g in gens] >= 0)
    @constraint(model, sum(α[g] for g in gens) == 1)
    @variable(model, δ[l in lines] >= 0)

    @variable(model, 0 <= M1[g in gens, k in 1:K, s in 1:S] <= 1)
    @variable(model, 0 <= M2[g in gens, k in 1:K, s in 1:S] <= 1)
    @variable(model, 0 <= M3[l in lines, k in 1:K, s in 1:S] <= 1)
    @variable(model, 0 <= M4[l in lines, k in 1:K, s in 1:S] <= 1)
    return (pBar=pBar, α=α, δ=δ, M1=M1, M2=M2, M3=M3, M4=M4)
end

function add_objective(model::Model, gens::Vector{String}, vars, gmm_Ω, c1::Dict, c2::Dict)
    EΩ = sum(gmm_Ω[:β][k]*gmm_Ω[:m][k] for k in 1:length(gmm_Ω[:β]))
    VΩ = sum(gmm_Ω[:β][k]*(gmm_Ω[:σ][k]^2 + gmm_Ω[:m][k]^2)
             for k in 1:length(gmm_Ω[:β])) - EΩ^2
    @objective(model, Min,
        sum(
            c2[g]*(vars.pBar[g] - vars.α[g]*EΩ)^2 +
            c2[g]*vars.α[g]^2*VΩ +
            c1[g]*(vars.pBar[g] - vars.α[g]*EΩ)
            for g in gens
        )
    )
end

function add_gen_constraints(model::Model, gens::Vector{String}, vars, gmm_Ω, pwl::Dict, ϵ::Float64)
    S = length(pwl[:a])
    for g in gens
        pmin = model[:_data]["gen"][g]["pmin"]
        pmax = model[:_data]["gen"][g]["pmax"]
        for k in 1:length(gmm_Ω[:β])
            μk, σk, βk = gmm_Ω[:m][k], gmm_Ω[:σ][k], gmm_Ω[:β][k]
            @constraint(model, vars.pBar[g] - μk*vars.α[g] <= pmax)
            @constraint(model, vars.pBar[g] - μk*vars.α[g] >= pmin)
            exprU = (pmax - vars.pBar[g] + μk*vars.α[g]) / σk
            @constraint(model, sum(βk*vars.M1[g,k,s] for s in 1:S) >= (1-ϵ)*vars.α[g])
            @constraint(model, [s=1:S],
                        vars.M1[g,k,s] <= pwl[:a][s]*exprU + pwl[:b][s]*vars.α[g]
            )
            exprL = (vars.pBar[g] - pmin - μk*vars.α[g]) / σk
            @constraint(model, sum(βk*vars.M2[g,k,s] for s in 1:S) >= (1-ϵ)*vars.α[g])
            @constraint(model, [s=1:S],
                        vars.M2[g,k,s] <= pwl[:a][s]*exprL + pwl[:b][s]*vars.α[g]
            )
        end
    end
end

function add_line_constraints(model::Model,
                              cont_gens::Vector{String},
                              parsed,
                              vars,
                              gmm_η::Dict{String,Dict},
                              pwl::Dict,
                              ϵ::Float64)
    S = length(pwl[:a])
    for (li, l) in enumerate(parsed.lines)
        rate = parsed.data["branch"][l]["rate_a"]
        γ = sum(parsed.hgen[li][parsed.gen_idx[g]] * vars.α[g] for g in cont_gens)

        cov_type = get(gmm_η[l], :cov_type, "tied")
        C0 = cov_type == "spherical" ? Matrix{Float64}(I,2,2) : gmm_η[l][:C0]

    
        L = cholesky(C0).L
        @constraint(model, [L*[γ;1]; vars.δ[l]] in SecondOrderCone())

        # PWL chance constraints for mixture components
        for k in 1:length(gmm_η[l][:λ])
            λk = gmm_η[l][:λ][k]
            νk = gmm_η[l][:ν][:,k]
            τk = gmm_η[l][:τ][k]

            expr = -sum(parsed.hgen[li][parsed.gen_idx[g]] * vars.pBar[g]
                        for g in cont_gens)
                   + parsed.f0_const[li]
                   + dot(νk, [γ,1])
            @constraint(model, -rate <= expr <= rate)

            # upper-tail PWL
            exprU = (rate - expr) / τk
            @constraint(model, sum(λk * vars.M3[l,k,s] for s in 1:S) >= (1-ϵ)*vars.δ[l])
            @constraint(model, [s=1:S],vars.M3[l,k,s] <= pwl[:a][s]*exprU + pwl[:b][s]*vars.δ[l])

            # lower-tail PWL
            exprL = (rate + expr) / τk
            @constraint(model, sum(λk * vars.M4[l,k,s] for s in 1:S) >= (1-ϵ)*vars.δ[l])
            @constraint(model, [s=1:S],vars.M4[l,k,s] <= pwl[:a][s]*exprL + pwl[:b][s]*vars.δ[l]) 
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
    vars   = add_variables(model, cont_gens, parsed.lines, K, S)

    @constraint(model,
        sum(vars.pBar[g] for g in cont_gens) + sum(parsed.p_wind_bar) == parsed.total_demand
    )

    add_objective(model,       cont_gens, vars,   gmm_Ω, c1, c2)
    add_gen_constraints(model, cont_gens, vars,   gmm_Ω, pwl, ϵ)
    #add_line_constraints(model,cont_gens, parsed, vars,   gmm_η, pwl, ϵ)

    return model
end
