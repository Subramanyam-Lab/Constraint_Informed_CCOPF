using PowerModels, JuMP, Gurobi, CSV, DataFrames, Distributions, LinearAlgebra

function read_xi_gaussian(path::String)
    df = CSV.read(path, DataFrame)
    return Dict(row.bus => (row.μ, row.σ^2) for row in eachrow(df))
end

function add_gen_constraints!(
    model::Model,
    data::Dict,
    pbar::Dict{String,VariableRef},
    α::Dict{String,VariableRef},
    xi_params::Dict{Int,Tuple{Float64,Float64}},
    ε::Float64
)
    μΩ   = sum(v[1] for v in values(xi_params))
    VarΩ = sum(v[2] for v in values(xi_params))
    σΩ   = sqrt(VarΩ)
    z    = quantile(Normal(), 1 - ε)

    for g in keys(pbar)
        pmin = data["gen"][g]["pmin"]
        pmax = data["gen"][g]["pmax"]
        @constraint(model, pbar[g] - α[g]*μΩ + z*α[g]*σΩ ≤ pmax)
        @constraint(model, pbar[g] - α[g]*μΩ - z*α[g]*σΩ ≥ pmin)
    end
end

function add_line_constraints!(
    model::Model,
    data::Dict,
    pbar::Dict{String,VariableRef},
    α::Dict{String,VariableRef},
    xi_params::Dict{Int,Tuple{Float64,Float64}},
    ε::Float64
)
    H = PowerModels.calc_basic_ptdf_matrix(data)
    buses = collect(keys(data["bus"]))              
    nb = length(buses)
    bus_idx = Dict(buses[i] => i for i in 1:nb)

    p_wind_bar = zeros(nb)
    for (g, gen) in data["gen"]
        if !haskey(pbar, g)
            bi = bus_idx[string(gen["gen_bus"])]
            p_wind_bar[bi] += gen["pg"]
        end
    end
    load_bus = zeros(nb)
    for (_, ld) in data["load"]
        bi = bus_idx[string(ld["load_bus"])]
        load_bus[bi] += ld["pd"]
    end
    net_base = p_wind_bar .- load_bus
    f0_const = H * net_base                     

    μΩ   = sum(v[1] for v in values(xi_params))
    VarΩ = sum(v[2] for v in values(xi_params))
    σΩ   = sqrt(VarΩ)
    z    = quantile(Normal(), 1 - ε)

    nl = size(H, 1)

    gens_ctrl = sort(collect(keys(pbar)))
    h_lg = Dict{Tuple{Int,String},Float64}()
    for (li, _) in enumerate(1:nl)
        for g in gens_ctrl
            bi = bus_idx[string(data["gen"][g]["gen_bus"])]
            h_lg[(li, g)] = -H[li, bi]
        end
    end

    @variable(model, t_line[li = 1:nl] >= 0)

    for li in 1:nl
        μ_loc = 0.0
        Var_loc = 0.0
        Cov_Ω_loc = 0.0
        for (bus_num, (μi, vari)) in xi_params
            sb = string(bus_num)
            haskey(bus_idx, sb) || continue
            col = bus_idx[sb]
            h = H[li, col]
            μ_loc    += h * μi
            Var_loc  += (h^2) * vari
            Cov_Ω_loc += h * vari          
        end

        m_l = [μΩ, μ_loc]
        C_l = [VarΩ  Cov_Ω_loc;
               Cov_Ω_loc  Var_loc]

        C_l = Symmetric(C_l)
        L = cholesky(C_l, check=true).L

        @expression(model, f_det, f0_const[li] - sum(h_lg[(li,g)] * pbar[g] for g in gens_ctrl))
        @expression(model, γ_l,   sum(h_lg[(li,g)] * α[g]    for g in gens_ctrl))

        #mean flow
        @expression(model, Efl, f_det + γ_l * m_l[1] + m_l[2])
        #varaince flow
        @expression(model, v_soc[1:2], [γ_l, 1.0])
        @expression(model, Lv[1:2], L * v_soc)
        @constraint(model, [t_line[li]; Lv] in SecondOrderCone())

        br_id = collect(keys(data["branch"]))[li]
        limit = get(data["branch"][br_id], "rate_a", 0.0)

        @constraint(model,  Efl + z * t_line[li] ≤  limit)
        @constraint(model,  Efl - z * t_line[li] ≥ -limit)
    end
end

function build_ccopf_gaussian(
    case_file::String,
    xi_csv::String,
    renewables::Vector{String},
    ε::Float64=0.05
)
    data = PowerModels.parse_file(case_file)

    # zero‐cost renewables
    for g in renewables
        data["gen"][g]["cost"] = [0.0, 0.0, 0.0]
    end

    all_g = collect(keys(data["gen"]))
    cond  = [g for g in all_g if data["gen"][g]["pg"] == 0.0]
    conv  = sort(setdiff(all_g, renewables, cond))

    xi_params = read_xi_gaussian(xi_csv)

    model = Model(Gurobi.Optimizer)

    @variable(model, pbar[g in conv] >= 0)
    @variable(model, α[g in conv]   >= 0)
    @constraint(model, sum(α[g] for g in conv) == 1)
    μΩ   = sum(v[1] for v in values(xi_params))
    VarΩ = sum(v[2] for v in values(xi_params))
    @objective(model, Min,
        sum(
            data["gen"][g]["cost"][1] * (pbar[g] - α[g]*μΩ)^2 +
            data["gen"][g]["cost"][2] * (pbar[g] - α[g]*μΩ) +
            data["gen"][g]["cost"][1] * α[g]^2 * VarΩ
        for g in conv)
    )
    total_wind = sum(data["gen"][w]["pg"] for w in renewables)
    total_load = sum(l["pd"] for l in values(data["load"]))
    @constraint(model,
        sum(pbar[g] for g in conv) + total_wind == total_load
    )

    add_gen_constraints!(model, data, pbar, α, xi_params, ε)
    add_line_constraints!(model, data, pbar, α, xi_params, ε)

    return model
end

function main()
    # paths
    case_file = "../data/c118swf.m"
    xi_csv     = "../data/xi_gauss.csv"
    renewables = ["5","11","12","25","28","29","30","37","45","51"]
    ε = 0.05
    model = build_ccopf_gaussian(case_file, xi_csv, renewables, ε)
    optimize!(model)

    println("Objective value: ", objective_value(model))
end

main()
