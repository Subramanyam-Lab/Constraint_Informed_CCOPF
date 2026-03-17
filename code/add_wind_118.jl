using PowerModels
using LinearAlgebra
using CSV
using DataFrames

# total load for wind penetration
function compute_total_load(data::Dict)::Float64
    return sum(data["load"][l]["pd"] for l in keys(data["load"]))
end

# MODIFIED: Instead of adding new IDs, we modify existing generators to be wind units
function convert_to_wind_generators!(
    data::Dict,
    wind_gen_ids::Vector{String},
    capacity_factor::Float64
)::Vector{String}

    for gen_id in wind_gen_ids
        # We assume wind units operate at a specific expected power (pg) 
        # based on their original pmax and the system's capacity factor.
        pmax = data["gen"][gen_id]["pmax"]
        pg   = capacity_factor * pmax

        data["gen"][gen_id]["pg"] = pg
        data["gen"][gen_id]["pmin"] = 0.0
        # Optional: Set cost to 0 for wind units
        data["gen"][gen_id]["cost"] = [0.0, 0.0]
        
        println("Converted Gen $gen_id to wind: pg = $pg, pmax = $pmax")
    end

    return wind_gen_ids
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
    # Configuration for the 118-bus case
    capacity_factor = 0.50
    # Your specified indices for wind units
    wind_indices = ["5", "11", "12", "25", "28", "29", "30", "37", "45", "51"]
    
    # Path to your 118-bus case file
    case_path = joinpath(@__DIR__, "..", "data", "c118swf.m") 

    data = PowerModels.make_basic_network(parse_file(case_path))

    total_load = compute_total_load(data)
    println("Total load = ", total_load)

    # Convert existing generators to wind
    wind_ids = convert_to_wind_generators!(
        data,
        wind_indices,
        capacity_factor
    )

    println("Wind generators: ", wind_ids)

    # Extract branch (line) ordering
    # Using sort to ensure consistent row ordering for the estimation script
    line_ids = sort(collect(keys(data["branch"])), by=x->parse(Int, x))  

    ptdf = compute_ptdf(data)
    println("PTDF dimension: ", size(ptdf))

    buses, bus_idx = build_bus_index(data)

    wind_bus_indices = get_wind_bus_indices(data, wind_ids, bus_idx)
    println("Wind bus indices (PTDF columns): ", wind_bus_indices)

    # Extract rows in the sorted line_id order
    H_full = ptdf
    line_indices = [findfirst(==(l), collect(keys(data["branch"]))) for l in line_ids]
    
    # H_wind should be [Lines x Wind_Buses]
    H_wind = H_full[:, wind_bus_indices]
    
    println("Hl submatrix dimension: ", size(H_wind))

    output_csv = joinpath(@__DIR__, "..", "data", "H_wind_118_matrix.csv")
    export_wind_ptdf(H_wind, line_ids, output_csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end