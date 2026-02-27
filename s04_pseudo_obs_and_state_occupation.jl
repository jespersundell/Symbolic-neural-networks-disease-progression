using DataFrames, CSV
using Statistics, Distributions, Random
#########################################################################
####################################################################
include("s00_simulator_functions.jl")


## extract pseudo observations
df = CSV.read("Data/Full_dataset_val.csv", DataFrame)
df = select(df, Not(:state))
rename!(df, :trans => :state)
df.start = firstdigit.(df.state)

time_point = 12+1
dt, πtobs = Aalen_Johansen_estimator(df, time_point)
πtobs12 = πtobs[:,time_point]

po = pseudo_observations(df, πtobs12, time_point)
a = normalize_pseudo_observations(po)

df_pseudo_obs = DataFrame(s1 = a[:,1],
                    s2 = a[:,2],
                    s3 = a[:,3],
                    s4 = a[:,4],
                    s5 = a[:,5])
#CSV.write("Data/pseudo_observations_validation_1years.csv", df_pseudo_obs)

occupancy_matrix_filtered, id_vec_filtered = state_occupancy(df, 12)

df_obs = DataFrame(ID = id_vec_filtered,
                            s1 = occupancy_matrix_filtered[:,1],
                            s2 = occupancy_matrix_filtered[:,2],
                            s3 = occupancy_matrix_filtered[:,3],
                            s4 = occupancy_matrix_filtered[:,4],
                            s5 = occupancy_matrix_filtered[:,5])
#CSV.write("Data/state_occupancy_validation.csv", df_obs)

