using Symbolics
using Lux, Zygote, Distributions, Optimisers, Random, MLUtils
using ComponentArrays
using CSV, DataFrames
using JLD2

@variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
include("s00_functions.jl")
include("s00_maskdense_objects.jl")

df4 = CSV.read("Data/S4.csv", DataFrame)
df4_test, df4 = train_test_data_split(df4, 0.1)

COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test  = prepare_data_for_NN_age_change(df4, df4_test)
COVS_new
NN_model = SNNmodel(10)

ps_λ5, ls5 = Lux.setup(rng, NN_model)
ps = ComponentArray{Float32}()
ps = ComponentArray(ps;ps_λ5)
opt = Adam(0.001)
opt_state = Optimisers.setup(opt, ps)

for jj in 1:1
    try
        iter = jj
        NN_model = SNNmodel(10)

        ps_λ5, ls5 = Lux.setup(rng, NN_model)
        ps = ComponentArray{Float32}()
        ps = ComponentArray(ps;ps_λ5)
        opt = Adam(0.001)
        opt_state = Optimisers.setup(opt, ps)

        ps = init_ps()

        println("Starting training of model $jj.")
        ######################################## training ##########################################
        ps, lossvec, lossvec_test = training_test_train(2500, 45, loss4, NN_model, state, ps, ls5,
                                    COVS_new, TIME, ID_new,
                                    state_test, COVS_new_test, TIME_test, ID_new_test)
        pruning_losses_train = []
        pruning_losses_test = []
        pruning_N_parameters = []
        pruning_parameter_index = []
        push!(pruning_losses_train, loss4(ps, ls5, state, COVS_new, TIME', NN_model, ID_new)[1])
        push!(pruning_losses_test, loss4(ps, ls5, state_test, COVS_new_test, TIME_test', NN_model, ID_new_test)[1])
        push!(pruning_N_parameters, count_parameters(ps))
        push!(pruning_parameter_index, 0)

        println("Starting pruning of model $jj.")
        for i in 1:1
            if i == 1
                ls5, Index_vector = zero_params(ps, ls5, state, COVS_new, TIME, NN_model, ID_new, 20)
            elseif i ==2 
                ls5, Index_vector = zero_params(ps, ls5, state, COVS_new, TIME, NN_model, ID_new, 10)
            elseif i == 3 || i ==4
                ls5, Index_vector = zero_params(ps, ls5, state, COVS_new, TIME, NN_model, ID_new, 5)        
            elseif i > 4
                ls5, Index_vector = zero_params(ps, ls5, state, COVS_new, TIME, NN_model, ID_new, 1)
            end
            parameters_left = count_parameters(ls5)
            ps ,_ ,_  = training_test_train(2000, 25, loss4, NN_model, state, ps, ls5,
                                                            COVS_new, TIME, ID_new,
                                                            state_test, COVS_new_test, TIME_test, ID_new_test)

            
            
            train_loss = loss4(ps, ls5, state, COVS_new, TIME', NN_model, ID_new)[1]
            test_loss = loss4(ps, ls5, state_test, COVS_new_test, TIME_test', NN_model, ID_new_test)[1]
            push!(pruning_losses_train, train_loss)
            push!(pruning_losses_test, test_loss)
            push!(pruning_N_parameters, parameters_left)
            push!(pruning_parameter_index, Index_vector[1])

            @save "Results/SNNmodels/mod$iter/mod_params$parameters_left.jld2" ps ls5
            if parameters_left <= 5
                break
            end
        end

        pruning_df = DataFrame(loss_train = pruning_losses_train,
                                loss_test = pruning_losses_test,
                                No_parameters = pruning_N_parameters,
                                parameter_index = pruning_parameter_index)
        CSV.write("Results/SNNmodels/mod$iter/SNN_pruning_summary.csv", pruning_df)
        @save "Results/SNNmodels/mod$iter/mod.jld2" ps ls5

        λ45_equation = getexpressions(ps.ps_λ5, ls5, true)
        expr = Meta.parse(λ45_equation)
        simple_expr = simplify(eval(expr))
        write("Results/SNNmodels/mod$iter/lambda45_rounded.txt", string(simple_expr))

        λ45_equation = getexpressions(ps.ps_λ5, ls5, false)
        expr = Meta.parse(λ45_equation)
        simple_expr = simplify(eval(expr))
        write("Results/SNNmodels/mod$iter/lambda45.txt", string(simple_expr))
        catch
            println("Error")
    end
end
