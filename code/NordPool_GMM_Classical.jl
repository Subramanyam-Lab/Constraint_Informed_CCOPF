using PowerModels, JuMP, LinearAlgebra, CSV, DataFrames, Gurobi, MathOptInterface, Statistics, Distributions, Random


function load_pwl_segments(path::String)
    df = CSV.read(path, DataFrame)
    return Dict(:a => Vector{Float64}(df.slope), :b => Vector{Float64}(df.intercept))
end

function load_gmm_10d(path::String)
    df = CSV.read(path, DataFrame)
    K = nrow(df)
    c_names = [Symbol("C_$(i)_$(j)") for i in 0:9 for j in 0:9]
    C0_vals = Vector(df[1, c_names]) 
    C0 = reshape(C0_vals, 10, 10)' 
    
    weights = df.weight
    sigmas = [df.tau2[k] .* C0 for k in 1:K]
    means = [zeros(10) for _ in 1:K] 
    
    return (weights=weights, means=means, covars=sigmas)
end

function parse_network(path::String, wind_ids::Vector{String})
    data = PowerModels.make_basic_network(PowerModels.parse_file(path))
    sbase = data["baseMVA"]
    
    buses = sort(collect(keys(data["bus"])), by=x->parse(Int,x))
    gens  = sort(collect(keys(data["gen"])), by=x->parse(Int,x))
    lines = sort(collect(keys(data["branch"])), by=x->parse(Int,x))

    nb, ng, nl = length(buses), length(gens), length(lines)
    bus_idx = Dict(buses[i] => i for i in 1:nb)
    gen_idx = Dict(gens[i] => i for i in 1:ng)

    H = PowerModels.calc_basic_ptdf_matrix(data)
    
    p_load = zeros(nb)
    for (k, d) in data["load"]; p_load[bus_idx[string(d["load_bus"])]] -= d["pd"]; end
    f_load = H * p_load 

    hgen = [zeros(ng) for _ in 1:nl]
    for (li, l) in enumerate(lines), (gi, g) in enumerate(gens)
        hgen[li][gi] = H[li, bus_idx[string(data["gen"][g]["gen_bus"])]]
    end

    wind_bus_indices = [bus_idx[string(data["gen"][w]["gen_bus"])] for w in wind_ids]
    H_wind = H[:, wind_bus_indices]

    return (data=data, gens=gens, lines=lines, gen_idx=gen_idx, bus_idx=bus_idx,
            H_wind=H_wind, H=H, hgen=hgen, f_load=f_load, sbase=sbase,
            total_demand=sum(d["pd"] for (k,d) in data["load"]))
end

function build_ccopf_10d(parsed, wind_ids, gmm, pwl, ϵ)
    cont = [g for g in parsed.gens if !(g in wind_ids)]
    K, S = length(gmm.weights), length(pwl[:a])
    ones_10 = ones(10)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, 0 <= pBar[g in cont] <= parsed.data["gen"][g]["pmax"])
    @variable(model, α[g in cont] >= 0)
    @constraint(model, sum(α[g] for g in cont) == 1.0)
    
    p_wind_nom = sum(parsed.data["gen"][w]["pg"] for w in wind_ids)
    @constraint(model, sum(pBar[g] for g in cont) + p_wind_nom == parsed.total_demand)

    @variable(model, M1[cont, 1:K] >= 0)
    @variable(model, M2[cont, 1:K] >= 0)
    for g in cont
        pmax, pmin = parsed.data["gen"][g]["pmax"], parsed.data["gen"][g]["pmin"]
        for k in 1:K
            σk = sqrt(ones_10' * gmm.covars[k] * ones_10)
            mk = ones_10' * gmm.means[k]
            zU = (pmax - pBar[g] + mk * α[g]) / σk
            zL = (pBar[g] - pmin - mk * α[g]) / σk
            for s in 1:S
                @constraint(model, M1[g,k] <= pwl[:a][s] * zU + pwl[:b][s] * α[g])
                @constraint(model, M2[g,k] <= pwl[:a][s] * zL + pwl[:b][s] * α[g])
            end
        end
        @constraint(model, sum(gmm.weights[k] * M1[g,k] for k in 1:K) >= (1-ϵ)*α[g])
        @constraint(model, sum(gmm.weights[k] * M2[g,k] for k in 1:K) >= (1-ϵ)*α[g])
    end

    @variable(model, δ[parsed.lines, 1:K] >= 0)
    @variable(model, M3[parsed.lines, 1:K] >= 0)
    @variable(model, M4[parsed.lines, 1:K] >= 0)

    for (li, l) in enumerate(parsed.lines)
        rate = parsed.data["branch"][l]["rate_a"]
        if rate <= 0.0 || rate >= 9.0 continue end
        
        f_gen = sum(parsed.hgen[li][parsed.gen_idx[g]] * pBar[g] for g in cont)
        f_wind = sum(parsed.hgen[li][parsed.gen_idx[w]] * parsed.data["gen"][w]["pg"] for w in wind_ids)
        f0 = parsed.f_load[li] + f_gen + f_wind
        γ = sum(parsed.hgen[li][parsed.gen_idx[g]] * α[g] for g in cont)

        for k in 1:K
            Σ10 = gmm.covars[k]
            h_l = parsed.H_wind[li, :]
            c11, c12, c22 = ones_10'Σ10*ones_10, ones_10'Σ10*h_l, h_l'Σ10*h_l
            Ck = [c11 c12; c12 c22]
            νk = [ones_10'gmm.means[k], h_l'gmm.means[k]]
            
            L = cholesky(Symmetric(Ck + 1e-9*I)).L
            @constraint(model, [δ[l,k]; L' * [-γ; 1.0]] in SecondOrderCone())

            numU = rate - f0 - (-γ * νk[1] + νk[2])
            numL = rate + f0 + (-γ * νk[1] + νk[2])

            for s in 1:S
                @constraint(model, M3[l,k] <= pwl[:a][s] * numU + pwl[:b][s] * δ[l,k])
                @constraint(model, M4[l,k] <= pwl[:a][s] * numL + pwl[:b][s] * δ[l,k])
            end
        end
        @constraint(model, sum(gmm.weights[k] * M3[l,k] for k in 1:K) >= (1-ϵ))
        @constraint(model, sum(gmm.weights[k] * M4[l,k] for k in 1:K) >= (1-ϵ))
    end

    @objective(model, Min, sum(parsed.data["gen"][g]["cost"][1] * pBar[g] for g in cont))
    return model
end

function run_oos_test_nordpool(parsed, wind_ids, pBar_opt, α_opt, seed)
    test_path = "../data/Nordtest_data_seed_$seed.csv"
    if !isfile(test_path)
        error("Holdout file not found: $test_path")
    end
    
    test_df = CSV.read(test_path, DataFrame)
    ξ = Matrix(test_df)' 
    
    N_wind, N_samples = size(ξ)
    Ω_samples = sum(ξ, dims=1)[:] 
    Λ_samples = parsed.H_wind * ξ

    cont = [g for g in parsed.gens if !(g in wind_ids)]
    vio_gen, vio_line = Dict(g => 0 for g in cont), Dict(l => 0 for l in parsed.lines)
    max_v = 0.0

    for (li, l) in enumerate(parsed.lines)
        rate = parsed.data["branch"][l]["rate_a"]
        if rate <= 0.0 || rate >= 9.0 continue end 

        γ_l = sum(parsed.hgen[li][parsed.gen_idx[g]] * α_opt[g] for g in cont)
        f_gen = sum(parsed.hgen[li][parsed.gen_idx[g]] * pBar_opt[g] for g in cont)
        f_wind = sum(parsed.hgen[li][parsed.gen_idx[w]] * parsed.data["gen"][w]["pg"] for w in wind_ids)
        f_base = parsed.f_load[li] + f_gen + f_wind

        for i in 1:N_samples
            f_act = f_base + Λ_samples[li, i] - γ_l * Ω_samples[i]
            if abs(f_act) > rate + 1e-6
                vio_line[l] += 1
                max_v = max(max_v, abs(f_act) - rate)
            end
        end
    end

    for i in 1:N_samples, g in cont
        p_act = pBar_opt[g] - α_opt[g] * Ω_samples[i]
        if p_act > parsed.data["gen"][g]["pmax"] + 1e-6 || p_act < parsed.data["gen"][g]["pmin"] - 1e-6
            vio_gen[g] += 1
            max_v = max(max_v, p_act - parsed.data["gen"][g]["pmax"], parsed.data["gen"][g]["pmin"] - p_act)
        end
    end

    rho_vals = [v/N_samples for v in values(vio_line)]
    append!(rho_vals, [v/N_samples for v in values(vio_gen)])
    return maximum(rho_vals, init=0.0), max_v
end

function main()
    wind_ids = ["1", "3", "4", "11", "17", "19", "35", "36", "50", "54"]
    pwl = load_pwl_segments("pwl_segments.csv")
    
    summary_df = DataFrame(seed=Int[], max_rho=Float64[])

    for s in 42:51
 
        gmm = load_gmm_10d("../data/gmm118Nord_Classical_gmm_$s.csv")
        parsed = parse_network("../data/c118swf.m", wind_ids)
        
        for (i, branch) in parsed.data["branch"]
            if haskey(branch, "rate_a") && branch["rate_a"] > 0
                branch["rate_a"] *= 1.03
            end
        end
        model = build_ccopf_10d(parsed, wind_ids, gmm, pwl, 0.05)
        optimize!(model)

        st = solve_time(model)
        status = termination_status(model)

        if status == MOI.OPTIMAL
            cont = [g for g in parsed.gens if !(g in wind_ids)]
            p_opt = Dict(g => value(model[:pBar][g]) for g in cont)
            α_opt = Dict(g => value(model[:α][g]) for g in cont)
                        m_rho, mv = run_oos_test_nordpool(parsed, wind_ids, p_opt, α_opt, s)
            
            push!(summary_df, (s, m_rho))
        else
            push!(summary_df, (s, NaN))
        end
    end

    CSV.write("../data/118_classical_NordPool_optimization_summary.csv", summary_df)
end

main()