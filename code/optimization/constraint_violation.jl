
using PowerModels
using JuMP
using Gurobi
using LinearAlgebra
using Random
using MathOptInterface
using CSV, DataFrames

include("gmm_reformulation.jl")  

function read_omega_gmm_csv(path::String)
    df = CSV.read(path, DataFrame)
    β = Vector{Float64}(df.beta)
    m = Vector{Float64}(df.m)
    σ = Vector{Float64}(df.sigma)
    return Dict(:β=>β, :m=>m, :σ=>σ)
end

function read_xi_gaussian(path::String)
    df = CSV.read(path, DataFrame)
    return Dict(Int(row.bus) => (Float64(row.μ), Float64(row.σ)^2) for row in eachrow(df))
end

function read_xi_samples_csv(path::String)
    df = CSV.read(path, DataFrame)
    sample_cols = names(df, Not(:bus))
    samples = Dict{Int,Vector{Float64}}()
    for row in eachrow(df)
        samples[Int(row.bus)] = [Float64(row[c]) for c in sample_cols]
    end
    T = length(sample_cols)
    return samples, T
end

function extract_solution_pbar_alpha(model::Model)
    pbar = Dict{String,Float64}()
    α    = Dict{String,Float64}()
    for v in all_variables(model)
        nm = name(v)
        if startswith(nm, "pbar[")
            key = strip(nm[6:end-1], ['"', '\''])
            pbar[key] = value(v)
        elseif startswith(nm, "α[") || startswith(nm, "alpha[")
            lb = findfirst('[', nm)
            rb = lastindex(nm)
            key = strip(nm[lb+1:rb-1], ['"', '\''])
            α[key] = value(v)
        end
    end
    return pbar, α
end

function compute_worst_case_violations(parsed, cont_gens::Vector{String},
                                       pbar_sol::Dict{String,Float64},
                                       α_sol::Dict{String,Float64},
                                       xi_samples::Dict{Int,Vector{Float64}})
    H = parsed.H
    buses = parsed.buses
    bus_idx = parsed.bus_idx
    nl = length(parsed.lines)

    # deterministic part computation
    p_wind_bar = parsed.p_wind_bar
    load_bus = zeros(length(buses))
    for (_, ld) in parsed.data["load"]
        bi = bus_idx[string(ld["load_bus"])]
        load_bus[bi] += ld["pd"]
    end
    net_base = p_wind_bar .- load_bus
    f0_const = H * net_base

    h_lg = Dict{Tuple{Int,String},Float64}()
    for (li, _) in enumerate(1:nl)
        for g in cont_gens
            bi = bus_idx[string(parsed.data["gen"][g]["gen_bus"])]
            h_lg[(li,g)] = -H[li, bi]
        end
    end

    line_limit = Dict{String,Float64}()
    for (l, br) in parsed.data["branch"]
        line_limit[l] = raw = get(br, "rate_a", 0.0)
    end

    pmin = Dict(g => parsed.data["gen"][g]["pmin"] for g in cont_gens)
    pmax = Dict(g => parsed.data["gen"][g]["pmax"] for g in cont_gens)
    T = length(first(values(xi_samples)))

    function omega_t(t)
        s = 0.0
        for v in values(xi_samples)
            s += v[t]
        end
        return s
    end

    gamma = zeros(nl)
    for li in 1:nl
        gamma[li] = sum(h_lg[(li,g)] * α_sol[g] for g in cont_gens)
    end

    gen_violation = Dict{String,Float64}(g => 0.0 for g in cont_gens)
    line_violation = Dict{String,Float64}(l => 0.0 for l in parsed.lines)

    for t in 1:T
        Ωt = omega_t(t)

        # Generators
        for g in cont_gens
            p_real = pbar_sol[g] - α_sol[g]*Ωt
            up = max(0.0, p_real - pmax[g])
            lo = max(0.0, pmin[g] - p_real)
            gen_violation[g] = max(gen_violation[g], max(up, lo))
        end

        # Lines
        ξ_bus_t = zeros(length(buses))
        for (busnum, samples) in xi_samples
            sb = string(busnum)
            haskey(bus_idx, sb) || continue
            ξ_bus_t[bus_idx[sb]] = samples[t]
        end
        local_term = H * ξ_bus_t

        for (li, l) in enumerate(parsed.lines)
            f_det  = f0_const[li] - sum(h_lg[(li,g)] * pbar_sol[g] for g in cont_gens)
            f_real = f_det + gamma[li]*Ωt + local_term[li]
            v = max(0.0, abs(f_real) - line_limit[l])
            line_violation[l] = max(line_violation[l], v)
        end
    end

    return Dict(
        :gen_max_violation    => maximum(values(gen_violation)),
        :gen_violation_per_g  => gen_violation,
        :line_max_violation   => maximum(values(line_violation)),
        :line_violation_per_l => line_violation,
    )
end

function main()
    Random.seed!(42)

    case_file         = "../data/c118swf.m"
    wind_idx          = ["5","11","12","25","28","29","30","37","45","51"]
    omega_gmm_csv     = "../data/omega_gmm.csv"      
    xi_gauss_csv      = "../data/xi_gauss.csv"       
    xi_samples_csv    = "../data/xi_samples.csv"    
    ε = 0.05

    # --- Load TRUE estimates ---
    gmm_Ω    = read_omega_gmm_csv(omega_gmm_csv)
    xi_param = read_xi_gaussian(xi_gauss_csv)        
    parsed_temp = parse_network(case_file, wind_idx)
    gmm_η = Dict{String,Dict}()   

    model = build_ccopf_constraint_informed(case_file, wind_idx, gmm_Ω, gmm_η,
                                            Dict(:a=>Float64[], :b=>Float64[]),
                                            ε)

    optimize!(model)
    println("Termination: ", termination_status(model))

    pbar_sol, α_sol = extract_solution_pbar_alpha(model)

    parsed    = parse_network(case_file, wind_idx)
    cont_gens = extract_controllable_generators(parsed.data, wind_idx)
    xi_samples, T = read_xi_samples_csv(xi_samples_csv)
    report = compute_worst_case_violations(parsed, cont_gens, pbar_sol, α_sol, xi_samples)

    println("gen:",report[:gen_max_violation])
    println("line:",report[:line_max_violation])

end

main()
