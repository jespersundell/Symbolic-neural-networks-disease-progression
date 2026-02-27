using DataFrames, CSV
using Statistics, Distributions, Random
using DifferentialEquations

include("s04_pseudo_obs_and_state_occupation.jl")

################## PREDICTIONS ################################################
df = CSV.read("Data/S1_val.csv", DataFrame)
df = select(df, Not(:Tl, :Tu, :time, :state, :trans))
covs_norm = covariate_matrix_pred(df)

covs = back_transform_covariates(covs_norm)

pred_mat = prediction_matrix(covs, 1)

df_pred = DataFrame(s1 = pred_mat[:,1],
                    s2 = pred_mat[:,2],
                    s3 = pred_mat[:,3],
                    s4 = pred_mat[:,4],
                    s5 = pred_mat[:,5])
#####################################################################################
############# VALIDATION ############################################################
### Calculate KL divergence
# df_pseudo_obs = CSV.read("Data/pseudo_observations_validation_1years.csv", DataFrame)
# df_pred = CSV.read("Data/Kunina_predictions.csv", DataFrame)
# df_obs = CSV.read("Data/state_occupancy_validation.csv", DataFrame)

Kunina_KL_divergence = KL_divergence(df_pseudo_obs, df_pred)

mean(Kunina_KL_divergence)
median(Kunina_KL_divergence)
mode(Kunina_KL_divergence)

obs_id = df_obs.ID
df_obs = select(df_obs, Not(:ID))

df_id = CSV.read("Data/S1_val.csv", DataFrame)
id_pred = df_id.ID
df_pred[!,:ID] =id_pred
df_pred

filtered_df_pred = filter(row -> row.ID in obs_id, df_pred)
filtered_df_pred = select(filtered_df_pred, Not(:ID))

Kunina_BS_observed = Brier_score(df_obs, filtered_df_pred)
mean(Kunina_BS_observed)
median(Kunina_BS_observed)
mode(Kunina_BS_observed)
########################
## fill up the BS obs Vector
# diff = length(Kunina_BS) - length(Kunina_BS_observed)

# BS_obs_full = vcat(Kunina_BS_observed, fill(99, diff) )

# performance_df = DataFrame(KL = Kunina_KL_divergence,
#                             BS_pseudo = Kunina_BS,
#                             BS_obs = BS_obs_full)

# CSV.write("Data/Kunina_performance_validation_1years.csv", performance_df)