using PowerModels
using LinearAlgebra
using CSV
using DataFrames

function compute_total_load(data::Dict)::Float64
    return sum(data["load"][l]["pd"] for l in keys(data["load"]))
end

function get_largest_generator_ids(data::Dict, n::Int)::Vector{String}
    gens = collect(keys(data["gen"]))
    sorted = sort(gens, by = g -> data["gen"][g]["pmax"], rev = true)
    return sorted[1:n]
end

function add_wind_generators!(
    data::Dict,
    wind_host_gens::Vector{String},
    wind_capacity_per_farm::Float64,
    capacity_factor::Float64
)::Vector{String}

    existing_ids = parse.(Int, collect(keys(data["gen"])))
    next_id = maximum(existing_ids) + 1

    wind_ids = String[]

    for (i, host_gen) in enumerate(wind_host_gens)

        gen_id = string(next_id + i - 1)
        push!(wind_ids, gen_id)

        bus_id = data["gen"][host_gen]["gen_bus"]

        pmax = wind_capacity_per_farm
        pg   = capacity_factor * pmax

        data["gen"][gen_id] = Dict(
            "gen_bus" => bus_id,
            "pg"      => pg,
            "qg"      => 0.0,
            "pmax"    => pmax,
            "pmin"    => 0.0,
            "qmax"    => 0.0,
            "qmin"    => 0.0,
            "vg"      => 1.0,
            "status"  => 1,
            "cost"    => [0.0, 0.0]
        )
    end

    return wind_ids
end

function build_bus_index(data::Dict)
    buses = sort(collect(keys(data["bus"])))
    nb = length(buses)
    bus_idx = Dict(buses[i] => i for i in 1:nb)
    return buses, bus_idx
end

function compute_ptdf(data::Dict)
    return PowerModels.calc_basic_ptdf_matrix(data)
end

function get_wind_bus_indices(
    data::Dict,
    wind_ids::Vector{String},
    bus_idx::Dict
)::Vector{Int}

    wind_bus_indices = Int[]

    for g in wind_ids
        bus_id = string(data["gen"][g]["gen_bus"])
        push!(wind_bus_indices, bus_idx[bus_id])
    end

    return wind_bus_indices
end

function export_wind_ptdf(
    H_wind::Matrix,
    line_ids::Vector{String},
    output_path::String
)

    df = DataFrame(H_wind, :auto)
    rename!(df, Symbol.(string.(1:size(H_wind,2))))

    df.line_id = line_ids
    select!(df, Cols(:line_id, Not(:line_id)))

    CSV.write(output_path, df)
    println("Wind PTDF matrix exported to: ", output_path)
end

function main()

    renewable_penetration = 0.20
    n_wind = 50
    capacity_factor = 0.50

    case_path = joinpath(@__DIR__, "..", "data", "case2746wp.m")

    data = PowerModels.make_basic_network(parse_file(case_path))

    total_load = compute_total_load(data)
    wind_total_capacity = renewable_penetration * total_load
    wind_capacity_per_farm = wind_total_capacity / n_wind

    wind_host_gens = get_largest_generator_ids(data, n_wind)
    wind_ids = add_wind_generators!(
        data,
        wind_host_gens,
        wind_capacity_per_farm,
        capacity_factor
    )

    line_ids = collect(keys(data["branch"]))  

    ptdf = compute_ptdf(data)
    println("PTDF dimension: ", size(ptdf))

    buses, bus_idx = build_bus_index(data)

    wind_bus_indices = get_wind_bus_indices(data, wind_ids, bus_idx)
    H_wind = ptdf[:, wind_bus_indices]
    output_csv = joinpath(@__DIR__, "..", "data", "H_wind_matrix.csv")
    export_wind_ptdf(H_wind, line_ids, output_csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end