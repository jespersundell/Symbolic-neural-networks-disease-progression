using DataFrames, CSV
using Statistics, Distributions, Random
using Lux

include("s04_pseudo_obs_and_state_occupation.jl")
################## PREDICTIONS ################################################
λ12(icovs, AGE) = sigmoid( -7.349414447202377 + 3.3382179379095525*AGE^2 ) 
λ13(icovs, AGE) = sigmoid( -3.770404886735833 + -4.413718147566824(1.6138322353363037*AGE)^(-2.2612500190734863*AGE) )
λ15(icovs, T, AGE) = sigmoid( -5.088961009701693 + 2.934063205129912*T + 3.960568589280301*AGE - 3.6709020089887474((1.0280190706253052*AGE)^(-1.3238271474838257*AGE)) )

λ24(icovs, AGE) = sigmoid( -13.693691391701853 + 9.075794440658983*AGE )
λ25(icovs, T, AGE) = sigmoid( -11.640552708141586 + 5.867275753606777*AGE + 5.558278024114554*T*AGE )

λ34(icovs, AGE) = sigmoid( -5.876191139221191 ) 
λ35(T, AGE) = sigmoid( -4.332710620699743 / ((abs(-0.3409731686115265*T - 0.7926619648933411*AGE)^1.9785043001174927)*(1 + abs(0.32811596989631653 / (abs(-0.3409731686115265*T - 0.7926619648933411*AGE)^1.9785043001174927)))) )

λ45(icovs, T, AGE) = sigmoid( -12.797488 + 4.2819874019999995*T + 8.4688195284*AGE ) 


df = CSV.read("Data/S1_val.csv", DataFrame)
df = select(df, Not(:Tl, :Tu, :time, :state, :trans))
covs = covariate_matrix_pred(df)

pred_mat = prediction_matrix_SNN(covs, 12)

df_pred = DataFrame(s1 = pred_mat[:,1],
                    s2 = pred_mat[:,2],
                    s3 = pred_mat[:,3],
                    s4 = pred_mat[:,4],
                    s5 = pred_mat[:,5])
#CSV.write("Data/SNN_predictions_validation_1years.csv", df_pred)
#####################################################################################
############# VALIDATION ############################################################
# df_pseudo_obs = CSV.read("Data/pseudo_observations_validation.csv", DataFrame)
# df_pred = CSV.read("Data/SNN_predictions_validation_1years.csv", DataFrame)
# df_obs = CSV.read("Data/state_occupancy_validation.csv", DataFrame)

SNN_KL_divergence = KL_divergence(df_pseudo_obs, df_pred)
mean(SNN_KL_divergence)
median(SNN_KL_divergence)
mode(SNN_KL_divergence)
### Calculate Brier score based on observations
obs_id = df_obs.ID
df_obs = select(df_obs, Not(:ID))

df_id = CSV.read("Data/S1_val.csv", DataFrame)
id_pred = df_id.ID
df_pred[!,:ID] =id_pred

filtered_df_pred = filter(row -> row.ID in obs_id, df_pred)
filtered_df_pred = select(filtered_df_pred, Not(:ID))

SNN_BS_observed = Brier_score(df_obs, filtered_df_pred)
mean(SNN_BS_observed)
median(SNN_BS_observed)
mode(SNN_BS_observed)
########################
## fill up the BS obs Vector
# diff = length(SNN_BS) - length(SNN_BS_observed)

# BS_obs_full = vcat(SNN_BS_observed, fill(99, diff) )

# performance_df = DataFrame(KL = SNN_KL_divergence,
#                             BS_pseudo = SNN_BS,
#                             BS_obs = BS_obs_full)

#CSV.write("Data/SNN_performance_validation_1years.csv", performance_df)
# ##########################################################################