using DataFrames, CSV
using Statistics, Distributions, Random
using Plots
using ExpectationMaximization
using Lux

include("s00_simulator_functions.jl")
include("s02_Mixture_model.jl")


###########################################################################
### Load datasets ########
###########################################################################
df_covs = CSV.read("Data/S1_val.csv", DataFrame)
df_covs = select(df_covs, Not(:Tl, :Tu, :time, :state, :trans))
COVS = covariate_matrix(df_covs)

df = CSV.read("Data/Full_dataset_val.csv", DataFrame)
df = select(df, Not(:state))
rename!(df, :trans => :state)
#df = df[1:100, :]
###########################################################################
###########################################################################
### AJ of observed data ########
###########################################################################
max_time = maximum(df.time)
dt, πtobs = Aalen_Johansen_estimator(df, max_time)
###########################################################################

###########################################################################
### AJ of simulated data ########
###########################################################################
iterations = 100 ## Number of simulations for VPC
## simulation data
sum_sim = AJ_summary_stats(COVS, max_time, iterations, mixture_fit)
###########################################################################

## For raw data only
p1 = AJ_plot(dt, πtobs, "Non-parametric probability.")
## For VPC-like figure without CIs
p1 = VPC_plot(dt, πtobs, sum_sim, "VPC (N=$iterations).")
## For VPC-like figure with CIs
s1, s2, s3, s4, s5 = bootstrap_raw_data(df, 1000)

s1sum = bootstrap_summary(s1)
s2sum = bootstrap_summary(s2)
s3sum = bootstrap_summary(s3)
s4sum = bootstrap_summary(s4)
s5sum = bootstrap_summary(s5)

pcomb = VPC_bootstrap_plot(s1sum, s2sum, s3sum, s4sum, s5sum, sum_sim, "VPC (N=$iterations).")

# savefig(p1, "Results/VPC_val_right_censoring.png")

# CSV.write("Data/VPC_right_censoring_dataset_1000.csv", sum_sim)
