using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using CSV, DataFrames
#using Plots
using JLD2
include("s00_functions.jl")
include("s00_maskdense_objects.jl")

rng = Random.default_rng()
Random.seed!(1)
################################################################################################################
df1 = CSV.read("Data/S1.csv", DataFrame)
df1 = df1[1:200, :] ## Reduce dataset for testing. If you want to evaluate on the full dataset, remove this line of code
df1_test, df1 = train_test_data_split(df1, 0.1)

COVS, TIME, state, COVS_test, TIME_test, state_test = prepare_data_for_NN(df1, df1_test)
#################################################################################################################
#################################################################################################################
#################################################################################################################
################################################################################################
################################################################################################
NN_model = SNNmodel(9)
NN_model_time = SNNmodel(10)

ps_λ2, ls2 = Lux.setup(rng, NN_model)
ps_λ3, ls3 = Lux.setup(rng, NN_model)
ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
ps = ComponentArray{Float32}()
ps = ComponentArray(ps;ps_λ2)
ps = ComponentArray(ps;ps_λ3)
ps = ComponentArray(ps;ps_λ5)
opt = Adam(0.001)
opt_state = Optimisers.setup(opt, ps)

ps = init_ps()

ps ,_ ,_ = training_test_train(2500, 35, loss1_fn, NN_model, NN_model_time, state, ps, ls2, ls3, ls5,
                            COVS, TIME,
                            state_test, COVS_test, TIME_test)

                
###############################################################################################
############## Pruning ######################################################################
###############################################################################################
println("Initiating pruning...")

ls2, ls3, ls5, pruning_df = network_pruning(ps, ls2, ls3, ls5, state, COVS, TIME, NN_model, NN_model_time,  loss1_parameter_pruning,
                        state_test, COVS_test, TIME_test,
                        10, 2)

println("Pruning step done!")

