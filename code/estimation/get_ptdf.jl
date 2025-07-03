using PowerModels, CSV, DataFrames

network_data = PowerModels.parse_file("../data/c118swf.m")
H = PowerModels.calc_basic_ptdf_matrix(network_data)
#print(size(H))

H_df   = DataFrame(Matrix(H), :auto)                   
CSV.write("../data/ptdf_matrix.csv", H_df)             
