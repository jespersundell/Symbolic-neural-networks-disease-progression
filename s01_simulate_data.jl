using DataFrames, CSV
using Statistics, Distributions, Random
using Plots
using ExpectationMaximization, Lux

Random.seed!(100)

include("s00_simulator_functions.jl")


######### Simulate a training dataset
df_sim = simulate_full_dataset(30000, 115)
df1 = filter_data_state(df_sim, 1)
df2 = filter_data_state(df_sim, 2)
df3 = filter_data_state(df_sim, 3)
df4 = filter_data_state(df_sim, 4)


CSV.write("Data/Full_dataset.csv", df_sim)
CSV.write("Data/S1.csv", df1)
CSV.write("Data/S2.csv", df2)
CSV.write("Data/S3.csv", df3)
CSV.write("Data/S4.csv", df4)

######### Simulate a validation dataset
df_sim = simulate_full_dataset(10000, 115)
df1 = filter_data_state(df_sim, 1)
df2 = filter_data_state(df_sim, 2)
df3 = filter_data_state(df_sim, 3)
df4 = filter_data_state(df_sim, 4)


CSV.write("Data/Full_dataset_val.csv", df_sim)
CSV.write("Data/S1_val.csv", df1)
CSV.write("Data/S2_val.csv", df2)
CSV.write("Data/S3_val.csv", df3)
CSV.write("Data/S4_val.csv", df4)


# df= simulate_full_dataset(10000, 115)
# names(df)
# df.time = df.Tu
# df = select(df, Not(:state))
# rename!(df, :trans => :state)
# df = select(df, [:ID, :Tl, :Tu, :state, :time])

# max_time = maximum(df.time)
# dt, πtobs = Aalen_Johansen_estimator(df, max_time)
# p1 = AJ_plot(dt, πtobs, "Non-parametric probability training data.")