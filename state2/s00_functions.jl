
###########################################################
####### Functions ########################################
#########################################################
function train_test_data_split(df, frac)
    state = df.state
    unique_state = unique(state)
    Number_of_groups = length(unique_state)
    Nobs_grouped_state = similar(unique_state)

    for i in eachindex(unique_state)
        Nobs_grouped_state[i] = (count(state .== unique_state[i]) )
    end 
    Nobs_group_testdata = Int.(round.(frac .* Nobs_grouped_state ) )

    tot = 0.0
    accumulating_Nobs = zeros(length(Nobs_grouped_state))
    for i in eachindex(Nobs_grouped_state)
        tot += Nobs_grouped_state[i]
        accumulating_Nobs[i] = tot
    end
    accumulating_Nobs = Int.(accumulating_Nobs)

    Index_vector = []
    for i in 1:Number_of_groups
        if i ==1
            sample_group = sample(1:accumulating_Nobs[i], Nobs_group_testdata[i], replace=false)
        elseif i > 1 
            sample_group = sample(accumulating_Nobs[i-1]+1:accumulating_Nobs[i], Nobs_group_testdata[i], replace=false)
        end
        push!(Index_vector, sample_group)
    end
    Index_vector = reduce(vcat, Index_vector)

    testdata = df[Index_vector, :]
    traindata = df[Not(Index_vector), :]

    return testdata, traindata
end

function covariate_transformation_age_change(TIME, COVS, state)
    upper = TIME[:,2]
    time_vector = []
    id_vec = []
    state_vec = []
    max_time = maximum(upper)

    ## New for age chaning as inputs
    upper_months = upper ./12
    max_age = maximum(upper_months .+ COVS[9,:])

    for i in eachindex(upper)
        timevec = collect(0:1:upper[i])' ./max_time ## added normalization here
        covs_i = hcat(fill(COVS[1:end-1,i], length(timevec))...)
        timevec_months = collect(0:1:upper[i]) ./12 ## New for age chaning as inputs
        covs_i[9,:] = (covs_i[9,:] .+ timevec_months) ./ max_age ## New for age chaning as inputs
        # if i == 1
        #     println(covs_i[9,:])
        #     println(timevec_months)
        COVS_new = vcat(covs_i, timevec)
        ids = fill(i, length(timevec))
        push!(time_vector, COVS_new)
        push!(id_vec, ids)

        new_state = fill(state[i], length(timevec))
        push!(state_vec, new_state)
    end

    COVS_new = reduce(hcat, time_vector)
    ID_new = reduce(vcat, id_vec)
    state_new = reduce(vcat, state_vec)

    return COVS_new, ID_new, state_new   
end

function prepare_data_for_NN_age_change(df, df_test)
    ################################################################################################
    #################################### TRAINING DATA ################################################
    covs = zeros(8, length(df.state))
    covdf = df[:,7:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    TIME = hcat(df.Tl, df.Tu)
    state = df.state
    Nid = size(TIME)[1]
    AGE = df.age_mid #./ maximum(df.age_mid)
    COVS = hcat(AGE...)
    COVS = vcat(covs, COVS)
    T2 = hcat(zeros(Nid)...)
    COVS = vcat(COVS, T2)
    ID = collect(1:1:Nid)
    COVS_new, ID_new, state_new = covariate_transformation_age_change(TIME, COVS, state)
    ################################################################################################
    #################################### TEST DATA ################################################
    covs = zeros(8, length(df_test.state))
    covdf = df_test[:,7:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    TIME_test = hcat(df_test.Tl, df_test.Tu)
    state_test = df_test.state
    Nid_test = size(TIME_test)[1]
    AGE = df_test.age_mid #./ maximum(df_test.age_mid)
    COVS_test = hcat(AGE...)
    COVS_test = vcat(covs, COVS_test)
    T2 = hcat(zeros(Nid_test)...)
    COVS_test = vcat(COVS_test, T2)
    ID_test = collect(1:1:Nid_test)
    COVS_new_test, ID_new_test, state_new_test = covariate_transformation_age_change(TIME_test, COVS_test, state_test)

    COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test
end

function init_ps()
    NN_model = SNNmodel(9)
    NN_model_time = SNNmodel(10)

    ps_λ4, ls4 = Lux.setup(rng, NN_model)
    ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
    ps = ComponentArray{Float32}()
    ps = ComponentArray(ps;ps_λ4)
    ps = ComponentArray(ps;ps_λ5)
    loss = loss23(ps, ls4, ls5, state, COVS_new, TIME', NN_model, NN_model_time, ID_new)[1]
    println("Training loss is: ", loss)

    while isinf(loss) || loss > 45000
        NN_model = SNNmodel(9)
        NN_model_time = SNNmodel(10)
    
        ps_λ4, ls4 = Lux.setup(rng, NN_model)
        ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
        ps = ComponentArray{Float32}()
        ps = ComponentArray(ps;ps_λ4)
        ps = ComponentArray(ps;ps_λ5)
        loss = loss23(ps, ls4, ls5, state, COVS_new, TIME', NN_model, NN_model_time, ID_new)[1]
        println("Training loss is: ", loss)
        if !isinf(loss) && loss < 45000
            println("Training loss is: ", loss)
            break
        end
    end
    
    return ps
end

function loss23(params, ls4, ls5, state, X, T,  model4, model5, idvec)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]

    λ4pred = sigmoid(model4(X[1:9,:], params.ps_λ4, ls4)[1])
    λ5pred = sigmoid(model5(X, params.ps_λ5, ls5)[1])

    for i in eachindex(lower)
        λ4pred_i = λ4pred[idvec.==i]
        λ5pred_i = λ5pred[idvec.==i]
        if any((λ4pred_i + λ5pred_i) .> 0.998)
            #λ5pred_i = λ5pred_i .- eps()
            λ4pred_i = λ4pred_i * (1- maximum(λ5pred_i) )
        end


        # if λ4pred_i[1] < 0.00000000001
        #     λ4pred_i = fill(eps(), length(λ4pred_i) )
        # end
        ####################################################################################    
        if (state[i] != 99 ) && (lower[i] == upper[i]) ## we know the exact time of event
            survival = sum(log.(1 .- (λ5pred_i[1:end-1] .+λ4pred_i[1:end-1] )))
            if (state[i] == 4)
                loss += log(λ4pred_i[end]) + survival
            elseif (state[i] == 5)
                loss += log(λ5pred_i[end]) + survival
            end
        #######################################################################################
        elseif (state[i] == 99) ## right censored λ4
            survival = sum(log.(1 .- (λ5pred_i[1:end-1] .+λ4pred_i[1:end-1] )))
            loss += survival
        # ###########################################################################################
        elseif (state[i] != 99) && (lower[i] != upper[i])## interval censored λ4 ->5
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()
            for jj in (lower_i+1):upper_i
                λ4_time_i = λ4pred_i[1:jj] ## just to make sure λ4 and λ5 are of equal lengths
                λ5_time_i = λ5pred_i[1:jj]
                survival = prod(1 .- (λ5_time_i .+λ4_time_i) )
                if (state[i] == 4)
                    interval_prob += survival * λ4pred_i[end]
                elseif (state[i]==5)
                    interval_prob += survival * λ5pred_i[end]
                end
            end
            loss += log(interval_prob)
    #######################################################################################
        end
    end

    return -1*loss , 1
end

function training_test_train(epochs, patience, lossfn::Function, model4, model5, state, parameters, ls4, ls5, COVS, TIME, ID,
                            state_test, COVS_test, TIME_test, ID_test
                            ; opt_state = opt_state)
    lossvec = []
    lossvec_test = []
    push!(lossvec,lossfn(parameters, ls4, ls5, state, COVS, TIME', model4,model5, ID)[1] )
    push!(lossvec_test,lossfn(parameters, ls4, ls5, state_test, COVS_test, TIME_test', model4,model5, ID_test)[1] )
    t = time()

    ## for early stopping
    #patience = 10 # Number of epochs to wait before stopping
    min_delta = 0.01 # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    best_loss = Inf
    for epoch in 1:epochs
        (loss, _), back = pullback(lossfn, parameters, ls4,ls5, state, COVS, TIME',  model4,model5, ID)# ## updated to train_loader (x=COVS, y=TIME)
        grad, _ = back((one(loss), nothing))

        ####################################################################
        ## required since Julia was updated on the cluster
        flatgrad, regrad = destructure(grad)
        flatgrad_new = ifelse.(isnan.(flatgrad), 0.0f0, flatgrad)
        grad = regrad(flatgrad_new)
        #####################################################################
        opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
        losscount=loss

        if epoch % 5 == 0
            losscount = lossfn(parameters, ls4,ls5, state, COVS, TIME', model4,model5, ID)[1]
            losscount_test = lossfn(parameters, ls4,ls5, state_test, COVS_test, TIME_test', model4,model5, ID_test)[1]
            dt = (time() - t) /60
            println("Epoch: $epoch, Loss Train: $losscount")
            println("Epoch: $epoch, Loss Test: $losscount_test")
            println("Time elapsed: $dt min")
            println("-------------------------------------")
        end
        if epoch % 1 == 0
            loss_train = lossfn(parameters, ls4,ls5, state, COVS, TIME', model4,model5, ID)[1]
            loss_test = lossfn(parameters, ls4,ls5, state_test, COVS_test, TIME_test', model4,model5, ID_test)[1]
            push!(lossvec, loss_train)
            push!(lossvec_test, loss_test)
        end

        ## early stopping
        if epoch >50
            if loss_test < best_loss - min_delta
                best_loss = loss_test
                patience_counter = 0
            else
                patience_counter += 1
                if patience_counter >= patience
                    println("########################################")
                    println("#### INCREASING TEST LOSS ##############")
                    println("#### STOPPING TRAINING #################")
                    println("########################################")
                    break
                end
            end
        end

        ## NaN stopping
        if isnan(lossvec[end]) || isinf(lossvec[end])
            print("Loss is NaN. Stopping training!")
            break
        end

        ## Time stopping
        dt = (time() - t) /60
        if dt > 180
            print("Time is out. Stopping training!")
            @save "Results/SNNmodels/mod_SNN_full_time_limit.jld2" ps ls4 ls5
            break
        end
    end
    return parameters, lossvec, lossvec_test   
end

function prediction_df(COVS::Matrix, TIME::Matrix)
    T2 = TIME[:,2]
    Nid = size(COVS)[2]
    patient_matrix_4 = DataFrame(zeros(maximum(T2)+1, Nid), :auto)
    patient_matrix_5 = DataFrame(zeros(maximum(T2)+1, Nid), :auto)
    max_time = maximum(T2)    
    time_in_state = collect(0:1:max_time)' ./ max_time

    for i in 1:Nid
        individual_covs = hcat(fill(COVS[1:9,i], length(time_in_state))...)
        individual_input_time = vcat(individual_covs, time_in_state)
        ipred4 = sigmoid.( vec(NN_model(individual_covs, ps.ps_λ4, ls4)[1] ) )
        ipred5 = sigmoid.( vec(NN_model_time(individual_input_time, ps.ps_λ5, ls5)[1] ) )
        patient_matrix_4[!,i] = ipred4
        patient_matrix_5[!,i] = ipred5
    end

    return patient_matrix_4, patient_matrix_5
end
#####################################################################################################
######## PARAMETER PRUNING ###########################
####################################################################################################
### To use the hessian or diaghessian functions from Zygote.jl,
## the parameters need to be in a vector. Therefore, use a 
## special loss function which take the "flat" vector of parameters
## and converts them into a parameter object used in a Lux model
## before calculating the loss
function loss23_parameter_pruning(params, params_prune, Network_pruned, ls4, ls5, state, X, T,  model4, model5, idvec, re=re)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]
    params_prune = re(params_prune)

    if Network_pruned == 4
        λ4pred = sigmoid(model4(X[1:9,:], params_prune, ls4)[1])
        λ5pred = sigmoid(model5(X, params, ls5)[1])
    elseif Network_pruned == 5
        λ4pred = sigmoid(model4(X[1:9,:], params, ls4)[1])
        λ5pred = sigmoid(model5(X, params_prune, ls5)[1])
    end

    for i in eachindex(lower)
        λ4pred_i = λ4pred[idvec.==i]
        λ5pred_i = λ5pred[idvec.==i]
        if any((λ4pred_i + λ5pred_i) .> 0.999)
            λ4pred_i = λ4pred_i * (1- maximum(λ5pred_i) )
        end
        ####################################################################################    
        if (state[i] != 99 ) && (lower[i] == upper[i]) ## we know the exact time of event
            survival = sum(log.(1 .- (λ5pred_i[1:end-1] .+λ4pred_i[1:end-1] )))
            if (state[i] == 4)
                loss += log(λ4pred_i[end]) + survival
            elseif (state[i] == 5)
                loss += log(λ5pred_i[end]) + survival
            end
        #######################################################################################
        elseif (state[i] == 99) ## right censored λ4
            survival = sum(log.(1 .- (λ5pred_i[1:end-1] .+λ4pred_i[1:end-1] )))
            loss += survival
        ###########################################################################################
        elseif (state[i] != 99) && (lower[i] != upper[i])## interval censored λ4 ->5
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()
            for jj in (lower_i+1):upper_i
                λ4_time_i = λ4pred_i[1:jj] ## just to make sure λ4 and λ5 are of equal lengths
                λ5_time_i = λ5pred_i[1:jj]
                survival = prod(1 .- (λ5_time_i .+λ4_time_i) )
                if (state[i] == 4)
                    interval_prob += survival * λ4pred_i[end]
                elseif (state[i]==5)
                    interval_prob += survival * λ5pred_i[end]
                end
            end
            loss += log(interval_prob)
    ######################################################################################
        end
    end

    return -1*loss , 1
end

## Calculate salience of model parameters: θ^2 * diagonal element of hessian
function salience_parameters(params, params_prune, Network_pruned, ls4, ls5, state, COVS, TIME, model4, model5, ID)
    Nid = size(TIME)[1]
    flat, re = destructure(params_prune)
    diagonalhess = diaghessian(p -> loss23_parameter_pruning(params, p, Network_pruned, ls4, ls5, state, COVS, TIME', model4, model5, ID, re)[1], flat)[1]
    param_salience = (flat.^2) .* abs.(diagonalhess)
    return param_salience ./ Nid
end

function custom_argmin(arr::Vector)
    min_val = Inf  # Initialize with positive infinity
    min_index = 0

    for (index, value) in enumerate(arr)
        if value != 0 && value < min_val
            min_val = value
            min_index = index
        end
    end
    return min_index
end

function check_connection(covnumber::Int, COVS, flat_ls, re_ls)
    id1 = COVS[1:9]
    id1_check = COVS[1:9]

    ls5 = re_ls(flat_ls)

    if covnumber == 10
        testcov = hcat(vcat(id1, [0]) )
        testcov2 = hcat(vcat(id1_check, [10]) )
        output1 = NN_model_time(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model_time(testcov2, ps.ps_λ5, ls5)[1][1]
    elseif covnumber == 9
        id1_check[9] = 10
        testcov = hcat(vcat(id1, [0]) )
        testcov2 = hcat(vcat(id1_check, [0]) )
        output1 = NN_model_time(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model_time(testcov2, ps.ps_λ5, ls5)[1][1]
    elseif covnumber == 1
        id1_check[1] = 10
        testcov = hcat(vcat(id1, [0]) )
        testcov2 = hcat(vcat(id1_check, [0]) )
        output1 = NN_model_time(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model_time(testcov2, ps.ps_λ5, ls5)[1][1]
    end
    
    return output1 ≈ output2
end

function zero_params(params, params_prune, Network_pruned, ls4, ls5, state, COVS, TIME, model4, model5, ID, parameters_to_remove)
    ## set lowest salience parameter to 0.0
    param_salience = salience_parameters(params, params_prune, Network_pruned, ls4, ls5, state, COVS, TIME, model4, model5, ID)

    if any(isnan.(param_salience))
        println("Loss is NaN, stopping pruning.")
    end
    max_sal = maximum(param_salience)

    if Network_pruned == 4
        flatls, re_ls = destructure(ls4)
    elseif Network_pruned == 5
        flatls, re_ls = destructure(ls5)
    end

    Index_vector = Int32[]
    for i in 1:parameters_to_remove
        index = custom_argmin(param_salience)
        param_salience[index] = 0.0
        #####################################################
        ## check if connection still holds for covs 1, 9 and 10
        #####################################################
        if Network_pruned != 5
            flatls[index] = 0.0

        elseif Network_pruned ==5
            flatls[index] = 0.0

            #check1 = check_connection(1, COVS, flatls, re_ls)
            check9 = check_connection(9, COVS, flatls, re_ls)
            check10 = check_connection(10, COVS, flatls, re_ls)

            if check9 == true || check10 == true
                flatls[index] = 1.0
                param_salience[index] = max_sal ## to ensure parameter stays when removing disconnected parameters
                println("reinstating parameter $index.")
            end

            counter = 0.0
            while check9 == true || check10 == true
                println("Time, age or sex is removed, testing other parameter.")
                counter += 1.0

                println("While loop iteration No $counter.")
                index = custom_argmin(param_salience)
                param_salience[index] = 0.0
                flatls[index] = 0.0

                #check1 = check_connection(1, COVS, flatls, re_ls)
                check9 = check_connection(9, COVS, flatls, re_ls)
                check10 = check_connection(10, COVS, flatls, re_ls)

                if check9 == true || check10 == true
                    flatls[index] = 1.0
                    param_salience[index] = max_sal ## to ensure parameter stays when removing disconnected parameters
                    println("reinstating parameter $index.")
                end


               ## stopping loop
                if counter == 10.0
                    println("Counter maximized. Break loop.")
                    break
                elseif check9 == false && check10 == false
                    println("Covariates are connected.")
                    break
                end
            end
            #####################################################
            ## check if connection still holds for covs 1, 9 and 10
            #####################################################
        end


        push!(Index_vector, index)
    end


    ## set all 0.0 salience parameters (i.e. disconected weights and biases) to 0.0
    for i in eachindex(param_salience)
        if param_salience[i] == 0.0
            flatls[i] = 0.0
        end
    end

    layer_states = re_ls(flatls)

    return layer_states, Index_vector
end

function count_parameters(ps)
    flatps,_ = destructure(ps)
    return count(flatps .!=0.0)    
end

function network_pruning(ps, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new,
                        state_test, COVS_new_test, TIME_test, ID_new_test,
                        number_of_final_parameters4, number_of_final_parameters5)
    pruning_losses_train = []
    pruning_losses_test = []
    pruning_N_parameters4 = []
    pruning_N_parameters5 = []
    pruning_parameter_index4 = []
    pruning_parameter_index5 = []
    push!(pruning_losses_train, loss23(ps, ls4, ls5, state, COVS_new, TIME', model4, model5, ID_new)[1])
    push!(pruning_losses_test, loss23(ps, ls4, ls5, state_test, COVS_new_test, TIME_test', model4, model5, ID_new_test)[1])
    push!(pruning_N_parameters4, count_parameters(ls4))
    push!(pruning_N_parameters5, count_parameters(ls5))
    push!(pruning_parameter_index4, [0])
    push!(pruning_parameter_index5, [0])

    for i in 1:50
        println("Pruning step number $i.")
        if i == 1
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 20)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 20)  ## pruning λ5
        elseif i ==2 
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 6)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 6)  ## pruning λ5
        elseif i ==3 || i ==4 || i == 5
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 4)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 4)  ## pruning λ5
        elseif i ==6 || i ==7 
            ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 2)  ## pruning λ4
            ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 2)  ## pruning λ5
        elseif i > 7
            parameters_left4 = count_parameters(ls4)
            parameters_left5 = count_parameters(ls5)
            if (parameters_left4 > number_of_final_parameters4) && (parameters_left5 > number_of_final_parameters5)
                ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ4
                ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ5
            elseif (parameters_left4 > number_of_final_parameters4) && (parameters_left5 <= number_of_final_parameters5)
                Index_vector5 = [0]
                ls4, Index_vector4 = zero_params(ps.ps_λ5, ps.ps_λ4, 4, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ4
            elseif (parameters_left4 <= number_of_final_parameters4) && (parameters_left5 > number_of_final_parameters5)
                Index_vector4 = [0]
                ls5, Index_vector5 = zero_params(ps.ps_λ4, ps.ps_λ5, 5, ls4, ls5, state, COVS_new, TIME, model4, model5, ID_new, 1)  ## pruning λ5
            end
        end

        parameters_left4 = count_parameters(ls4)
        parameters_left5 = count_parameters(ls5)
        ps ,_ ,_ = training_test_train(1500, 25, loss23, model4, model5, state, ps, ls4, ls5,
                                    COVS_new, TIME, ID_new,
                                    state_test, COVS_new_test, TIME_test, ID_new_test)

        
        
        train_loss = loss23(ps, ls4, ls5, state, COVS_new, TIME', model4, model5, ID_new)[1]
        test_loss = loss23(ps, ls4, ls5, state_test, COVS_new_test, TIME_test', model4, model5, ID_new_test)[1]
        push!(pruning_losses_train, train_loss)
        push!(pruning_losses_test, test_loss)
        push!(pruning_N_parameters4, parameters_left4)
        push!(pruning_N_parameters5, parameters_left5)
        push!(pruning_parameter_index4, Index_vector4)
        push!(pruning_parameter_index5, Index_vector5)


        @save "Results/SNNmodels/mod_4params$parameters_left4-5params$parameters_left5.jld2" ps ls4 ls5
        if (parameters_left4 <= number_of_final_parameters4) && (parameters_left5 <= number_of_final_parameters5)
            break
        end
    end

    pruning_df = DataFrame(loss_train = pruning_losses_train,
                        loss_test = pruning_losses_test,
                        No_parameters4 = pruning_N_parameters4,
                        No_parameters5 = pruning_N_parameters5,
                        parameter_index4 = pruning_parameter_index4,
                        parameter_index5 = pruning_parameter_index5)

    return ls4, ls5, pruning_df#pruning_losses_train, pruning_losses_test, pruning_N_parameters4, pruning_N_parameters5, pruning_parameter_index4, pruning_parameter_index5
end
####################################################################
##############################################################################
## FUNCTIONS TO EXTRACT EQUATIONS
#############################################################################
##############################################################################
function layer2string(params, masks, layer, input, roundvals=false)
    p = params
    m = masks

    if layer in [1 3 5] # Dense layers
        W = p.weight .* m.W_mask  # Parameter value of weight with mask
        B = p.bias .* m.b_mask   # Parameter value of bias with mask

        if roundvals==true # rounding for visibility
            W = round.(W, digits=2) # round for visibility
            B = round.(B, digits=2)
        end

        n_outputs = size(W, 1)
        n_inputs = size(W, 2)

        l_str = String[]
        for j = 1:n_outputs
            push!(l_str, "")
            for k = 1:n_inputs
                w0 = W[j, k]
                if w0 != 0 # weights
                    if w0 < 0
                        l_str[j] = "$(l_str[j])+($(w0))*($(input[k]))"
                    else
                        l_str[j] = "$(l_str[j])+$(w0)*($(input[k]))"
                    end
                end
            end
            b0 = B[j]
            if b0 != 0 # bias
                if b0 < 0
                    l_str[j] = "$(l_str[j])+($(b0))"
                else
                    l_str[j] = "$(l_str[j])+$(b0)"
                end
            end
            if layer == 5 # put abs on output
                l_str[j] = "($(l_str[j]))"
            end
        end
    elseif layer == 2 # Activation functions (here treated as specific layers)
        l_str = String[]
        for i in eachindex(input)
            if i == 2 # Multiplication
                if isempty(input[i]) || isempty(input[i+1]) || input[i] == "0" || input[i+1] == "0" # if either of the inputs are 0                    push!(l_str, "0")
                else
                    push!(l_str, "($(input[i]))*($(input[i+1]))")
                end
            elseif isempty(input[i])
                if i == 5 # power function exponent = 0
                    push!(l_str, "1") # x^0 = 1
                else
                    push!(l_str, "0")
                end
            elseif i == 1 # passthrough
                push!(l_str, "($(input[i]))")
            elseif i == 5 # power function
                push!(l_str, "(abs($(input[4])))^($(input[i]))")
            end
        end
    elseif layer == 4 # Activation functions (here treated as specific layers)
        l_str = String[]
        for i in eachindex(input)
            if i == 2 # Multiplication
                if isempty(input[i]) || isempty(input[i+1]) || input[i] == "0" || input[i+1] == "0" # if either of the inputs are 0
                    push!(l_str, "0")
                else
                    push!(l_str, "($(input[i]))*($(input[i+1]))")
                end
            elseif isempty(input[i])
                if i != 3
                    push!(l_str, "0")
                end
            elseif i == 1 # passthrough
                push!(l_str, "($(input[i]))")
            elseif i == 4 # division function
                #println("$(input[i+1])")
                if "$(input[i+1])" != ""
                    push!(l_str, "($(input[i]))/(abs($(input[i+1])) + 1)")
                else
                    push!(l_str, "($(input[i]))/($(input[i+1]) + 1)")
                end
            end
        end
    end
    return l_str
end

function getexpressions(params, ls, roundvals=false)
    input = ["x1"; "x2"; "x3"; "x4"; "x5"; "x6"; "x7"; "x8"; "x9"; "x10"] 
    Y1 = layer2string(params.layer_1, ls.layer_1, 1, input, roundvals)
    Y2 = layer2string(params.layer_1, ls.layer_1, 2, Y1, roundvals)
    Y3 = layer2string(params.layer_3, ls.layer_3, 3, Y2, roundvals)
    Y4 = layer2string(params.layer_3, ls.layer_3, 4, Y3, roundvals)
    Y = layer2string(params.layer_5, ls.layer_5, 5, Y4, roundvals)[1]
    
    return Y
end

####################################################################
##############################################################################
## FUNCTIONS USED TO SET UP MODEL
#############################################################################
##############################################################################
function activation_1(x)
    return [x[1, :] (x[2, :] .* x[3, :]) powerfunction.(x[4, :], x[5, :])]'
end

function activation_2(x)
    return [x[1, :] (x[2, :] .* x[3, :]) (x[4, :] ./ (abs.(x[5, :]) .+ one(eltype(x))))]'
end

# Special power function called by activation functions.
function powerfunction(x1, x2)
    z = zero(eltype(x1))
    if x1 ≈ z
        return z # Return zero if base is zero.
    else
        return abs.(x1) .^ x2 # |a|^b, to avoid for example (-0.5)^1.2
    end
end

## Neural network with Custom layers
function SNNmodel(N_covs)
    m = Chain(  DenseMaskLayer(N_covs, 5),
                x -> activation_1(x),
                DenseMaskLayer(3, 5),
                x -> activation_2(x),
                DenseMaskLayer(3, 1))
    return m
end
