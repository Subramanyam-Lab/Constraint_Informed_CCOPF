using PowerModels
using JuMP
using LinearAlgebra
using CSV
using DataFrames
using Gurobi
using MathOptInterface
using Distributions
using Random

######################################################
### Use this script to do the optimization of CC-OPF
### on Synthetic-C with constrainted-informd estimation
######################################################

###### ----- Load Parameters ----- ######
function load_pwl_segments(path::String)
    df = CSV.read(path, DataFrame)
    colnames = string.(names(df))
    slope_col = names(df)[findfirst(name -> occursin("slope", string(name)), names(df))]
    int_col   = names(df)[findfirst(name -> occursin("intercept", string(name)), names(df))]
    return Dict(:a => Vector{Float64}(df[!, slope_col]), :b => Vector{Float64}(df[!, int_col]))
end

function load_gmm_omega(path)
    df = CSV.read(path, DataFrame)
    df = df[df.approach .== "zero_mean", :]
    sort!(df, :component)
    return Dict(
        :β => Vector{Float64}(df.weight),
        :σ => sqrt.(Vector{Float64}(df.variance)),
        :m => zeros(nrow(df))
    )
end

function load_gmm_eta(path::String)
    df = CSV.read(path, DataFrame)
    sort!(df, [:line_id, :component])
    lines = unique(df.line_id)
    Kη = maximum(df.component)
    gmm = Dict{String,Dict}()
    for l in lines
        sub = df[df.line_id .== l, :]
        sort!(sub, :component)
        λ = Vector{Float64}(sub.weight)
        τ = zeros(Kη)
        Σ_list = Vector{Matrix{Float64}}(undef, Kη)
        for k in 1:Kη
            Σ = [sub.cov_11[k] sub.cov_12[k]; sub.cov_12[k] sub.cov_22[k]]
            Σ_list[k] = Σ
            τ[k] = sqrt(tr(Σ) / 2.0)
        end
        C0 = Σ_list[1] / (τ[1]^2)
        ν = zeros(2, Kη)
        for k in 1:Kη
            ν[:,k] = [sub.mean_Omega[k], sub.mean_Lambda[k]]
        end
        gmm[string(l)] = Dict(:λ => λ, :τ => τ, :ν => ν, :C0 => C0)
    end
    return gmm, Kη
end

function parse_network(data::Dict, wind_ids::Vector{String})
    buses = sort(collect(keys(data["bus"])), by=x->parse(Int,x))
    gens  = sort(collect(keys(data["gen"])), by=x->parse(Int,x))
    lines = sort(collect(keys(data["branch"])), by=x->parse(Int,x))
    nb, ng, nl = length(buses), length(gens), length(lines)
    bus_idx  = Dict(buses[i] => i for i in 1:nb)
    gen_idx  = Dict(gens[i]  => i for i in 1:ng)
    total_demand = sum(data["load"][l]["pd"] for l in keys(data["load"]))
    p_gen_bar = zeros(nb)
    for g in gens
        bi = bus_idx[string(data["gen"][g]["gen_bus"])]
        p_gen_bar[bi] += data["gen"][g]["pg"] 
    end
    p_wind_bar = zeros(nb)
    for g in wind_ids
        bi = bus_idx[string(data["gen"][g]["gen_bus"])]
        p_wind_bar[bi] += data["gen"][g]["pg"]
    end
    load_bus = zeros(nb)
    for ld in keys(data["load"])
        bi = bus_idx[string(data["load"][ld]["load_bus"])]
        load_bus[bi] += data["load"][ld]["pd"]
    end
    net_base = p_gen_bar + p_wind_bar - load_bus
    H = PowerModels.calc_basic_ptdf_matrix(data)
    f0_const = H * net_base 
    hgen = [zeros(ng) for _ in 1:nl]
    for (li, l) in enumerate(lines)
        for (gi, g) in enumerate(gens)
            bi = bus_idx[string(data["gen"][g]["gen_bus"])]
            hgen[li][gi] = H[li, bi] 
        end
    end
    return (data=data, buses=buses, gens=gens, lines=lines, gen_idx=gen_idx, total_demand=total_demand, p_wind_bar=p_wind_bar, hgen=hgen, f0_const=f0_const)
end

###### ----- CC-OPF Reformulation ----- ######
function build_model(parsed, wind_ids, gmmΩ, gmmη, Kη::Int, pwl, ϵ)
    cont = [g for g in parsed.gens if !(g in wind_ids)]
    K = length(gmmΩ[:β])
    S = length(pwl[:a])
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    @variable(model, 0 <= pBar[g in cont] <= parsed.data["gen"][g]["pmax"])
    @variable(model, α[g in cont] >= 0)
    @constraint(model, sum(α[g] for g in cont) == 1)
    @constraint(model, sum(pBar[g] for g in cont) + sum(parsed.p_wind_bar) == parsed.total_demand)
    VΩ = sum(gmmΩ[:β][k]*gmmΩ[:σ][k]^2 for k in 1:K)
    c1 = Dict(g => parsed.data["gen"][g]["cost"][2] for g in cont)
    c2 = Dict(g => parsed.data["gen"][g]["cost"][1] for g in cont)
    @objective(model, Min, sum(c2[g]*(pBar[g]^2 + α[g]^2*VΩ) + c1[g]*pBar[g] for g in cont))
    @variable(model, M1[g in cont, k in 1:K] >= 0)
    @variable(model, M2[g in cont, k in 1:K] >= 0)
    for g in cont
        pmin, pmax = parsed.data["gen"][g]["pmin"], parsed.data["gen"][g]["pmax"]
        for k in 1:K
            σk, mk = gmmΩ[:σ][k], gmmΩ[:m][k]
            zU = (pmax - pBar[g] + mk*α[g]) / σk
            for s in 1:S @constraint(model, M1[g,k] <= pwl[:a][s] * zU + pwl[:b][s] * α[g]) end
        end
        @constraint(model, sum(gmmΩ[:β][k] * M1[g,k] for k in 1:K) >= (1 - ϵ) * α[g])
        for k in 1:K
            σk, mk = gmmΩ[:σ][k], gmmΩ[:m][k]
            zL = (pBar[g] - pmin - mk*α[g]) / σk
            for s in 1:S @constraint(model, M2[g,k] <= pwl[:a][s] * zL + pwl[:b][s] * α[g]) end
        end
        @constraint(model, sum(gmmΩ[:β][k] * M2[g,k] for k in 1:K) >= (1 - ϵ) * α[g])
    end
    @variable(model, δ[l in parsed.lines, k in 1:Kη] >= 0)
    @variable(model, M3[l in parsed.lines, k in 1:Kη] >= 0)
    @variable(model, M4[l in parsed.lines, k in 1:Kη] >= 0)
    for (li, l) in enumerate(parsed.lines)
        γ = sum(parsed.hgen[li][parsed.gen_idx[g]] * α[g] for g in cont)
        rate = parsed.data["branch"][l]["rate_a"]
        flow0 = sum(parsed.hgen[li][parsed.gen_idx[g]] * pBar[g] for g in cont) + parsed.f0_const[li]
        for k in 1:Kη
            Ck = (gmmη[l][:τ][k]^2) .* gmmη[l][:C0] 
            L = cholesky(Symmetric(Ck)).L
            νk = gmmη[l][:ν][:, k]
            @constraint(model, [δ[l,k]; L' * [γ; 1]] in SecondOrderCone())
            numU = rate - flow0 - (γ * νk[1] + νk[2])
            numL = rate + flow0 + (γ * νk[1] + νk[2])
            for s in 1:S
                @constraint(model, M3[l,k] <= pwl[:a][s] * numU + pwl[:b][s] * δ[l,k])
                @constraint(model, M4[l,k] <= pwl[:a][s] * numL + pwl[:b][s] * δ[l,k])
            end
        end
        @constraint(model, sum(gmmη[l][:λ][k] * M3[l,k] for k in 1:Kη) >= (1 - ϵ))
        @constraint(model, sum(gmmη[l][:λ][k] * M4[l,k] for k in 1:Kη) >= (1 - ϵ))
    end
    return model
end

function reliability_test(parsed, wind_ids, pBar_opt, α_opt, seed)
    Random.seed!(seed)
    N_samples, γ_cauchy = 2000, 0.02 
    dist = Cauchy(0, γ_cauchy)
    ξ = rand(dist, length(wind_ids), N_samples)
    Ω_samples = sum(ξ, dims=1)[:] 
    H_full = PowerModels.calc_basic_ptdf_matrix(parsed.data)
    bus_lookup = Dict(b => i for (i, b) in enumerate(parsed.buses))
    wind_bus_indices = [bus_lookup[string(parsed.data["gen"][id]["gen_bus"])] for id in wind_ids]
    H_wind = H_full[:, wind_bus_indices]
    Λ_samples = H_wind * ξ 
    cont = [g for g in parsed.gens if !(g in wind_ids)]
    α_vec = [get(α_opt, g, 0.0) for g in parsed.gens]
    p_vec = [get(pBar_opt, g, 0.0) for g in parsed.gens]
    vio_gen, vio_line = zeros(length(cont)), zeros(length(parsed.lines))
    for i in 1:N_samples
        for (gi, g) in enumerate(cont)
            p_act = pBar_opt[g] - α_opt[g] * Ω_samples[i]
            if p_act > parsed.data["gen"][g]["pmax"] + 1e-6 || p_act < parsed.data["gen"][g]["pmin"] - 1e-6
                vio_gen[gi] += 1
            end
        end
        for (li, l) in enumerate(parsed.lines)
            rate = parsed.data["branch"][l]["rate_a"]
            if rate <= 0.0 continue end
            γ_l = dot(parsed.hgen[li], α_vec)
            flow0 = (dot(parsed.hgen[li], p_vec) + parsed.f0_const[li]) / 100.0
            if abs(flow0 - γ_l * Ω_samples[i] + Λ_samples[li, i]) > rate + 1e-6
                vio_line[li] += 1
            end
        end
    end
    return max(maximum(vio_gen/N_samples, init=0.0), maximum(vio_line/N_samples, init=0.0))
end

function main()
    case_path, ϵ = "../data/c118swf.m", 0.05
    wind_ids = ["5", "11", "12", "25", "28", "29", "30", "37", "45", "51"]
    pwl = load_pwl_segments("pwl_segments.csv")
    results = DataFrame(seed=Int[], obj=Float64[], max_rho=Float64[])
    
    for s in 42:51
        println("Processing Seed: $s")
        data = PowerModels.make_basic_network(PowerModels.parse_file(case_path))
        for g in wind_ids
            pmax = data["gen"][g]["pmax"]
            data["gen"][g]["pg"], data["gen"][g]["pmin"], data["gen"][g]["cost"] = 0.8*pmax, 0.0, [0.0, 0.0]
        end
        parsed = parse_network(data, wind_ids)
        gmmΩ = load_gmm_omega("../data/gmm118_omega_zeromean_$s.csv")
        gmmη, Kη = load_gmm_eta("../data/gmm118_zero_mean_eta_l_seed$s.csv")
        model = build_model(parsed, wind_ids, gmmΩ, gmmη, Kη, pwl, ϵ)
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            cont = [g for g in parsed.gens if !(g in wind_ids)]
            pBar_opt = Dict(g => value(model[:pBar][g]) for g in cont)
            α_opt = Dict(g => value(model[:α][g]) for g in cont)
            rho = reliability_test(parsed, wind_ids, pBar_opt, α_opt, s)
            push!(results, (s, objective_value(model), rho))
            println(" Seed $s: Obj = $(objective_value(model)), Rho = $rho")
        else
            println(" Seed $s failed.")
        end
    end
    CSV.write("../data/118_ci_Cauchy_optimization_summary.csv", results)
end

main()