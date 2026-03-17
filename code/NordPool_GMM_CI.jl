using PowerModels, JuMP, LinearAlgebra, CSV, DataFrames, Gurobi, MathOptInterface, Statistics

##############################################################
### Solve CC-OPF 10 times using the constraint_informed 
### estimated parameters, and perform oos constraint 
### violation using 10 different partitions of the 
### original dataset
#############################################################

function load_pwl_segments(path::String)
    df = CSV.read(path, DataFrame)
    return Dict(:a => Vector{Float64}(df.slope), :b => Vector{Float64}(df.intercept))
end

function load_gmm_omega_scaled(path, scale_factor)
    df = CSV.read(path, DataFrame)
    return Dict(
        :β => Vector{Float64}(df.weight),
        :σ => Vector{Float64}(sqrt.(df.variance)) .* scale_factor,
        :m => zeros(nrow(df))
    )
end

function load_gmm_eta_scaled(path::String, scale_factor)
    df = CSV.read(path, DataFrame)
    sort!(df, [:line_id, :component])
    lines = unique(df.line_id)
    Kη = maximum(df.component)
    gmm = Dict{String,Dict}()
    for l in lines
        sub = df[df.line_id .== l, :]
        Σ_list = [ [sub.cov_11[k] sub.cov_12[k]; sub.cov_12[k] sub.cov_22[k]] .* (scale_factor^2) for k in 1:Kη ]
        gmm[string(l)] = Dict(:λ => Vector{Float64}(sub.weight), :Σ => Σ_list)
    end
    return gmm, Kη
end

function parse_network(data::Dict, wind_ids::Vector{String})
    buses = sort(collect(keys(data["bus"])), by=x->parse(Int,x))
    gens  = sort(collect(keys(data["gen"])), by=x->parse(Int,x))
    lines = sort(collect(keys(data["branch"])), by=x->parse(Int,x))
    
    nb, ng, nl = length(buses), length(gens), length(lines)
    bus_idx = Dict(buses[i] => i for i in 1:nb)
    gen_idx = Dict(gens[i] => i for i in 1:ng)

    p_net_nodal = zeros(nb)
    for ld in keys(data["load"])
        p_net_nodal[bus_idx[string(data["load"][ld]["load_bus"])]] -= data["load"][ld]["pd"]
    end
    
    p_wind_total = 0.0
    for g in wind_ids
        p_wind_total += data["gen"][g]["pg"]
        bi = bus_idx[string(data["gen"][g]["gen_bus"])]
        p_net_nodal[bi] += data["gen"][g]["pg"]
    end

    H = PowerModels.calc_basic_ptdf_matrix(data)
    f0_const = H * p_net_nodal 

    hgen = [zeros(ng) for _ in 1:nl]
    for (li, l) in enumerate(lines)
        for (gi, g) in enumerate(gens)
            bi = bus_idx[string(data["gen"][g]["gen_bus"])]
            hgen[li][gi] = H[li, bi] 
        end
    end

    return (data=data, buses=buses, gens=gens, lines=lines, 
            gen_idx=gen_idx, total_demand=sum(data["load"][l]["pd"] for l in keys(data["load"])),
            p_wind_total=p_wind_total, hgen=hgen, f0_const=f0_const)
end

function build_ccopf_model(parsed, wind_ids, gmmΩ, gmmη, Kη, pwl, ϵ)
    cont = [g for g in parsed.gens if !(g in wind_ids)]
    KΩ = length(gmmΩ[:β]); S = length(pwl[:a])
    model = Model(Gurobi.Optimizer); set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, parsed.data["gen"][g]["pmin"] <= pBar[g in cont] <= parsed.data["gen"][g]["pmax"])
    @variable(model, α[g in cont] >= 0)
    @constraint(model, sum(α[g] for g in cont) == 1.0)
    @constraint(model, sum(pBar[g] for g in cont) + parsed.p_wind_total == parsed.total_demand)

    VΩ = dot(gmmΩ[:β], gmmΩ[:σ].^2)
    @objective(model, Min, sum(parsed.data["gen"][g]["cost"][1]*pBar[g]^2 + 
                               parsed.data["gen"][g]["cost"][1]*α[g]^2*VΩ + 
                               parsed.data["gen"][g]["cost"][2]*pBar[g] for g in cont))

    @variable(model, M1[cont, 1:KΩ] >= 0); @variable(model, M2[cont, 1:KΩ] >= 0)
    for g in cont
        pmax, pmin = parsed.data["gen"][g]["pmax"], parsed.data["gen"][g]["pmin"]
        for k in 1:KΩ
            zU = (pmax - pBar[g]) / gmmΩ[:σ][k]; zL = (pBar[g] - pmin) / gmmΩ[:σ][k]
            for s in 1:S
                @constraint(model, M1[g,k] <= pwl[:a][s]*zU + pwl[:b][s]*α[g])
                @constraint(model, M2[g,k] <= pwl[:a][s]*zL + pwl[:b][s]*α[g])
            end
        end
        @constraint(model, sum(gmmΩ[:β][k]*M1[g,k] for k in 1:KΩ) >= (1-ϵ)*α[g])
        @constraint(model, sum(gmmΩ[:β][k]*M2[g,k] for k in 1:KΩ) >= (1-ϵ)*α[g])
    end

    @variable(model, δ[parsed.lines, 1:Kη] >= 0)
    @variable(model, M3[parsed.lines, 1:Kη] >= 0); @variable(model, M4[parsed.lines, 1:Kη] >= 0)
    for (li, l) in enumerate(parsed.lines)
        rate = parsed.data["branch"][l]["rate_a"]
        if rate >= 9.0 continue end
        γ_expr = sum(parsed.hgen[li][parsed.gen_idx[g]] * α[g] for g in cont)
        f0_expr = sum(parsed.hgen[li][parsed.gen_idx[g]] * pBar[g] for g in cont) + parsed.f0_const[li]
        for k in 1:Kη
            Σk = gmmη[string(l)][:Σ][k]
            L = cholesky(Symmetric(Σk + 1e-9*I)).L
            @constraint(model, [δ[l,k]; L' * [γ_expr; 1.0]] in SecondOrderCone())
            for s in 1:S
                @constraint(model, M3[l,k] <= pwl[:a][s]*(rate - f0_expr) + pwl[:b][s]*δ[l,k])
                @constraint(model, M4[l,k] <= pwl[:a][s]*(rate + f0_expr) + pwl[:b][s]*δ[l,k])
            end
        end
        @constraint(model, sum(gmmη[string(l)][:λ][k]*M3[l,k] for k in 1:Kη) >= (1-ϵ))
        @constraint(model, sum(gmmη[string(l)][:λ][k]*M4[l,k] for k in 1:Kη) >= (1-ϵ))
    end
    return model
end

function run_oos_test(parsed, wind_ids, pBar_opt, α_opt, seed)
    test_df = CSV.read("../data/Nordtest_data_seed_$seed.csv", DataFrame)
    N_samples = nrow(test_df)
    
    P_forecast_pu = [parsed.data["gen"][w]["pmax"] for w in wind_ids]
    ξ = Matrix(test_df)' .* P_forecast_pu
    Ω_samples = sum(ξ, dims=1)[:] 
    
    H_full = PowerModels.calc_basic_ptdf_matrix(parsed.data)
    bus_lookup = Dict(b => i for (i, b) in enumerate(parsed.buses))
    wind_bus_indices = [bus_lookup[string(parsed.data["gen"][id]["gen_bus"])] for id in wind_ids]
    H_wind = H_full[:, wind_bus_indices]
    Λ_samples = H_wind * ξ 

    cont = [g for g in parsed.gens if !(g in wind_ids)]
    vio_gen = Dict(g => 0 for g in cont)
    vio_line = Dict(l => 0 for l in parsed.lines)

    max_violation_val = 0.0

    α_vec = [get(α_opt, g, 0.0) for g in parsed.gens]
    p_vec = [get(pBar_opt, g, 0.0) for g in parsed.gens]

    for i in 1:N_samples
        Ω = Ω_samples[i]
        
        for g in cont
            p_act = pBar_opt[g] - α_opt[g] * Ω
            if p_act > parsed.data["gen"][g]["pmax"] + 1e-6 || p_act < parsed.data["gen"][g]["pmin"] - 1e-6
                vio_gen[g] += 1
            end
        end
        
        for (li, l) in enumerate(parsed.lines)
            rate = parsed.data["branch"][l]["rate_a"]
            if rate <= 0.0 || rate >= 9.0 continue end 

            γ_l = dot(parsed.hgen[li], α_vec)
            flow0 = dot(parsed.hgen[li], p_vec) + parsed.f0_const[li]
            f_actual = flow0 - γ_l * Ω + Λ_samples[li, i]

            if abs(f_actual) > rate + 1e-6
                vio_line[l] += 1
                max_violation_val = max(max_violation_val, abs(f_actual) - rate)
            end
        end
    end

    rho_gen = [vio_gen[g]/N_samples for g in cont]
    rho_line = [vio_line[l]/N_samples for l in parsed.lines]
    
    max_rho = isempty(rho_gen) && isempty(rho_line) ? 0.0 : max(
        maximum(rho_gen, init=0.0), 
        maximum(rho_line, init=0.0)
    )
    
    vio_results = filter(x -> x[2] > 0, vio_line)
    sorted_vio_lines = sort(collect(vio_results), by=x->x[2], rev=true)
    if !isempty(sorted_vio_lines)
        for i in 1:min(3, length(sorted_vio_lines))
            println("  Line $(sorted_vio_lines[i][1]): rho = $(sorted_vio_lines[i][2]/N_samples)")
        end
    end
    return max_rho, max_violation_val 
end

function main()
    summary_df = DataFrame(seed=Int[], max_rho=Float64[])
    pwl = load_pwl_segments("pwl_segments.csv")
    wind_ids = ["1", "3", "4", "11", "17", "19", "35", "36", "50", "54"]

    for s in 42:51
        data = PowerModels.make_basic_network(PowerModels.parse_file("../data/c118swf.m"))

        if haskey(data["branch"], "159")
            data["branch"]["159"]["rate_a"] *= 2.5
        end

        for g in wind_ids
            data["gen"][g]["pg"] *= 1.0
            data["gen"][g]["cost"] = [0.0, 0.0, 0.0] 
        end
        
        parsed = parse_network(data, wind_ids)
        cont = [g for g in parsed.gens if !(g in wind_ids)]

        avg_wind_pu = mean([data["gen"][w]["pg"] for w in wind_ids])
        gmmΩ = load_gmm_omega_scaled("../data/gmm118Nord_omega_seed$s.csv", avg_wind_pu)
        gmmη, Kη = load_gmm_eta_scaled("../data/gmm118Nord_eta_l_seed$s.csv", avg_wind_pu)

        model = build_ccopf_model(parsed, wind_ids, gmmΩ, gmmη, Kη, pwl, 0.05)
        optimize!(model)
        st = solve_time(model)
        status = termination_status(model)

        if status == MOI.OPTIMAL
            p_opt = Dict(g => value(model[:pBar][g]) for g in cont)
            α_opt = Dict(g => value(model[:α][g]) for g in cont)
            
            m_rho, mv = run_oos_test(parsed, wind_ids, p_opt, α_opt, s)
            
            push!(summary_df, (s, m_rho))
        else
            push!(summary_df, (s, NaN))
        end

        CSV.write("../data/118_ci_NordPool_optimization_summary.csv", summary_df)
    end
end

main()