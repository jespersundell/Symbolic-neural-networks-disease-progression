
########################################################################################################
########################################################################################################
############ PRUNING ##############################################################################
########################################################################################################
using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using CSV, DataFrames
using JLD2
include("s00_functions.jl")
include("s00_maskdense_objects.jl")

rng = Random.default_rng()
Random.seed!(45)

df2 = CSV.read("Data/S2.csv", DataFrame)
df2 = df2[1:200, :] ## Reduce dataset for testing. If you want to evaluate on the full dataset, remove this line of code
df2_test, df2 = train_test_data_split(df2, 0.1)

COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test = prepare_data_for_NN_age_change(df2, df2_test)

println("Setting up model...")
NN_model = SNNmodel(9)
NN_model_time = SNNmodel(10)

ps_λ4, ls4 = Lux.setup(rng, NN_model)
ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
ps = ComponentArray{Float32}()
ps = ComponentArray(ps;ps_λ4)
ps = ComponentArray(ps;ps_λ5)
opt = Adam(0.001)
opt_state = Optimisers.setup(opt, ps)

ps = init_ps()

println("Initializing optimization...")
opt = Adam(0.005)
opt_state = Optimisers.setup(opt, ps)

## Initial training
ps ,lossvec ,_ = training_test_train(2000, 45, loss23, NN_model, NN_model_time, state, ps, ls4, ls5,
                            COVS_new, TIME, ID_new,
                            state_test, COVS_new_test, TIME_test, ID_new_test)

## Pruning
ls4, ls5, pruning_df  =network_pruning(ps, ls4, ls5, state, COVS_new, TIME, NN_model, NN_model_time, ID_new,
                        state_test, COVS_new_test, TIME_test, ID_new_test,
                        5, 5) ## change number of parameters == 5 to == 10

#CSV.write("Results/SNNmodels/SNN_pruning_summary.csv", pruning_df)
