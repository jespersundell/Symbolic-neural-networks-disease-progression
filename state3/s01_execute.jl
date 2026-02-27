#using Symbolics
using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using CSV, DataFrames
#using Plots
using JLD2


include("s00_functions.jl")
include("s00_maskdense_objects.jl")
rng = Random.default_rng()
Random.seed!(3)
#@variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
################################################################################################################
df3 = CSV.read("Data/S3.csv", DataFrame)
df3_test, df3 = train_test_data_split(df3, 0.1)
COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test = prepare_data_for_NN_age_change(df3, df3_test)
#################################################################################################################
#################################################################################################################
#################################################################################################################

NN_model = SNNmodel(9)
NN_model_time = SNNmodel(10)

ps_λ4, ls4 = Lux.setup(rng, NN_model)
ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
ps = ComponentArray{Float32}()
ps = ComponentArray(ps;ps_λ4)
ps = ComponentArray(ps;ps_λ5)
opt = Adam(0.001)
opt_state = Optimisers.setup(opt, ps)

function network_pruning(ps, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new,
                        state_test, COVS_new_test, TIME_test, ID_new_test,
                        number_of_final_parameters,number_of_final_parameters5, model_number)
    pruning_losses_train = []
    pruning_losses_test = []
    pruning_N_parameters4 = []
    pruning_N_parameters5 = []
    push!(pruning_losses_train, loss23(ps, ls4, ls5, state, COVS_new, TIME', model4, model5, ID_new)[1])
    push!(pruning_losses_test, loss23(ps, ls4, ls5, state_test, COVS_new_test, TIME_test', model4, model5, ID_new_test)[1])
    push!(pruning_N_parameters4, count_parameters(ls4))
    push!(pruning_N_parameters5, count_parameters(ls5))


    for i in 1:40
        println("Removing parameters...")
        if i == 1
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 20)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 20)  ## pruning λ5
        elseif i ==2
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 10)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 10)  ## pruning λ5
        elseif i == 3
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 5)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 5)  ## pruning λ5
        elseif i > 3
            parameters_left4 = count_parameters(ls4)
            parameters_left5 = count_parameters(ls5)
            if (parameters_left4 > number_of_final_parameters) && (parameters_left5 > number_of_final_parameters5)
                ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ4
                ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ5
            elseif (parameters_left4 > number_of_final_parameters) && (parameters_left5 <= number_of_final_parameters5)
                Index_vector5 = [0]
                ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ4
            elseif (parameters_left4 <= number_of_final_parameters) && (parameters_left5 > number_of_final_parameters5)
                Index_vector4 = [0]
                ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ5
            end
        end

        #ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ5
        parameters_left4 = count_parameters(ls4)
        parameters_left5 = count_parameters(ls5)
        println("###########################################")
        println("Starting training. No of parameters left:")
        println("Lambda 4: $parameters_left4 parameters")
        println("Lambda 5: $parameters_left5 parameters")
        println("###########################################")
        ps ,_ ,_ = training_test_train(2000, 25, loss23, model4, model5, state, ps, ls4, ls5,
                                    COVS_new, TIME, ID_new,
                                    state_test, COVS_new_test, TIME_test, ID_new_test)

        
        
        train_loss = loss23(ps, ls4, ls5, state, COVS_new, TIME', model4, model5, ID_new)[1]
        test_loss = loss23(ps, ls4, ls5, state_test, COVS_new_test, TIME_test', model4, model5, ID_new_test)[1]
        push!(pruning_losses_train, train_loss)
        push!(pruning_losses_test, test_loss)
        push!(pruning_N_parameters4, parameters_left4)
        push!(pruning_N_parameters5, parameters_left5)

        @save "Results/SNNmodels/mod$model_number/mod_5params$parameters_left4-5params$parameters_left5.jld2" ps ls4 ls5

        if (parameters_left4 <= number_of_final_parameters) && (parameters_left5 <= number_of_final_parameters5)
            println("Final parameter count:")
            println("Lambda34: $parameters_left4.")
            println("Lambda35: $parameters_left5.")
            break
        end
        
        # if (parameters_left5 <= number_of_final_parameters)
        #     break
        # end
    end
    

    pruning_df = DataFrame(loss_train = pruning_losses_train,
                        loss_test = pruning_losses_test,
                        No_parameters4 = pruning_N_parameters4,
                        No_parameters5 = pruning_N_parameters5)

    return ps, ls4, ls5, pruning_df
end

############################################################################
######### TRAINING AND PRUNING LOOP #####################################################
############################################################################

for jj in 1:1
    try
        iter = jj
        println("Setting up model...")
        NN_model = SNNmodel(9)
        NN_model_time = SNNmodel(10)

        ps_λ4, ls4 = Lux.setup(rng, NN_model)
        ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
        ps = ComponentArray{Float32}()
        ps = ComponentArray(ps;ps_λ4)
        ps = ComponentArray(ps;ps_λ5)
        opt = Adam(0.005)
        opt_state = Optimisers.setup(opt, ps)

        ps = init_ps()
        ######################################## training ##########################################
        println("Initialization done! Starting training.")

        ps, lossvec, lossvec_test = training_test_train(5000, 45, loss23, NN_model, NN_model_time, state, ps, ls4, ls5,
                                    COVS_new, TIME, ID_new,
                                    state_test, COVS_new_test, TIME_test, ID_new_test)
        @save "Results/SNNmodels/mod$iter/mod_full.jld2" ps ls4 ls5
        ############################################################################################################
        opt = Adam(0.001)
        opt_state = Optimisers.setup(opt, ps)
        println("Starting pruning...")
        ps, ls4, ls5, pruning_df  =network_pruning(ps, ls4, ls5, state, COVS_new, TIME, NN_model, NN_model_time, ID_new,
                        state_test, COVS_new_test, TIME_test, ID_new_test,
                        5, 5, iter)
                        
                        
        @save "Results/SNNmodels/mod$iter/mod.jld2" ps ls4 ls5
        CSV.write("Results/SNNmodels/mod$iter/summary_pruning.csv", pruning_df)
    catch
        println("Error")
    end
end


