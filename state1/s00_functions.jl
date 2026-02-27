
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

function covariate_transformation(TIME, COVS, state)
    upper = TIME[:,2]
    time_vector = []
    id_vec = []
    state_vec = []
    max_time = maximum(upper)
    for i in eachindex(upper)
        timevec = collect(0:1:upper[i])' ./max_time ## added normalization here
        covs_i = hcat(fill(COVS[1:end-1,i], length(timevec))...)
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

function prepare_data_for_NN(df, df_test)
    ################################################################################################
    #################################### TRAINING DATA ################################################
    covs = zeros(8, length(df.state))
    covdf = df[:,7:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    Tmax = maximum(vcat(df.Tu, df_test.Tu) )  ## new
    TIME = hcat(df.Tl, df.Tu)
    state = df.state
    #Nid = size(TIME)[1]
    AGE = df.age_mid ./ maximum(df.age_mid)
    COVS = hcat(AGE...)
    COVS = vcat(covs, COVS)
    T2 = hcat(df.Tu...) ./ Tmax
    COVS = vcat(COVS, T2)
    #ID = collect(1:1:Nid)
    #COVS_new, ID_new, _ = covariate_transformation(TIME, COVS, state)
    ################################################################################################
    #################################### TEST DATA ################################################
    covs = zeros(8, length(df_test.state))
    covdf = df_test[:,7:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    TIME_test = hcat(df_test.Tl, df_test.Tu)
    state_test = df_test.state
    #Nid_test = size(TIME_test)[1]
    AGE = df_test.age_mid ./ maximum(df_test.age_mid)
    COVS_test = hcat(AGE...)
    COVS_test = vcat(covs, COVS_test)
    T2 = hcat(df_test.Tu...) ./Tmax
    COVS_test = vcat(COVS_test, T2)
    #ID_test = collect(1:1:Nid_test)
    #COVS_new_test, ID_new_test, _ = covariate_transformation(TIME_test, COVS_test, state_test)

    #return COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test
    return COVS, TIME, state, COVS_test, TIME_test, state_test
end

function init_ps()
    NN_model = SNNmodel(9)
    NN_model_time = SNNmodel(10)

    ps_λ2, ls2 = Lux.setup(rng, NN_model)
    ps_λ3, ls3 = Lux.setup(rng, NN_model)
    ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
    ps = ComponentArray{Float32}()
    ps = ComponentArray(ps;ps_λ2)
    ps = ComponentArray(ps;ps_λ3)
    ps = ComponentArray(ps;ps_λ5)
    loss = loss1_fn(ps, ls2, ls3, ls5, state, COVS, TIME', NN_model, NN_model_time)[1]
    println("Training loss is: ", loss)

    while isinf(loss)
        NN_model = SNNmodel(9)
        NN_model_time = SNNmodel(10)
    
        ps_λ2, ls2 = Lux.setup(rng, NN_model)
        ps_λ3, ls3 = Lux.setup(rng, NN_model)
        ps_λ5, ls5 = Lux.setup(rng, NN_model_time)
        ps = ComponentArray{Float32}()
        ps = ComponentArray(ps;ps_λ2)
        ps = ComponentArray(ps;ps_λ3)
        ps = ComponentArray(ps;ps_λ5)
        loss = loss1_fn(ps, ls2, ls3, ls5, state, COVS, TIME', NN_model, NN_model_time)[1]
        println("Training loss is: ", loss)
        if !isinf(loss)
            println("Final training loss is: ", loss)
            break
        end
    end
    
    return ps
end

function training_test_train(epochs, patience, lossfn::Function, model23, model5, state, parameters, ls2, ls3, ls5, COVS, TIME,
                            state_test, COVS_test, TIME_test
                            ; opt_state = opt_state)
    lossvec = []
    lossvec_test = []
    push!(lossvec,lossfn(parameters, ls2, ls3, ls5, state, COVS, TIME', model23, model5)[1] )
    push!(lossvec_test,lossfn(parameters, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23,model5)[1] )
    t = time()

    ## for early stopping
    #patience = 10 # Number of epochs to wait before stopping
    min_delta = 0.01 # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    best_loss = Inf
    for epoch in 1:epochs
        (loss, _), back = pullback(lossfn, parameters, ls2, ls3, ls5, state, COVS, TIME',  model23, model5)# ## updated to train_loader (x=COVS, y=TIME)
        grad, _ = back((one(loss), nothing))

        opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
        losscount=loss

        if epoch % 20 == 0
            losscount = lossfn(parameters, ls2, ls3, ls5, state, COVS, TIME', model23, model5)[1]
            losscount_test = lossfn(parameters, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23, model5)[1]
            dt = (time() - t) /60
            println("Epoch: $epoch, Loss Train: $losscount")
            println("Epoch: $epoch, Loss Test: $losscount_test")
            println("Time elapsed: $dt min")
            println("-------------------------------------")
        end
        # if epoch % 1 == 0
        #     loss_train = lossfn(parameters, ls2, ls3, ls5, state, COVS, TIME', model23, model5)[1]
        #     loss_test = lossfn(parameters, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23, model5)[1]
        #     push!(lossvec, loss_train)
        #     push!(lossvec_test, loss_test)
        # end
        loss_test = lossfn(parameters, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23, model5)[1]
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

        ## NaN and Inf stopping
        if isnan(lossvec[end]) || isinf(lossvec[end])
            println("Stopping due to NaN or Inf.")
            break
        end

        ## Time stopping
        dt = (time() - t) /60
        if dt > 1470 ## 30 hours used for a batch script with 29 hours stopping time
            println("Stopping due to time limit.")
            break
        end
    end
    return parameters, lossvec, lossvec_test   
end

function loss1_fn(params, ls2, ls3, ls5, state, X, T,  model23, model5)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]
    Tmax = maximum(upper)

    λ2pred = sigmoid(model23(X[1:9,:], params.ps_λ2, ls2)[1])
    λ3pred = sigmoid(model23(X[1:9,:], params.ps_λ3, ls3)[1])
    λ5pred = sigmoid(model5(X, params.ps_λ5, ls5)[1])

    for i in eachindex(lower)
        λ2pred_i = λ2pred[i]
        λ3pred_i = λ3pred[i]
        λ5pred_i = λ5pred[i]
        if (λ2pred_i + λ3pred_i + λ5pred_i) > 0.999
            λ2pred_i = λ2pred_i * (1-λ5pred_i)
            λ3pred_i = λ3pred_i * (1-λ5pred_i) * (1-λ2pred_i)
        end

        #########################################################################################    
        if (state[i] != 99 ) && (lower[i] == upper[i]) ## we know the exact time of event 5
            timevec = collect(0:1:upper[i])' ./Tmax
            covs_i = hcat(fill(X[1:9,i], length(timevec))...)
            λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
            survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            if any(survival .< 0.00000000001)
                λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)) .* (1-λ2pred_i) #.* (1-λ2pred_i)
                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            end

            if λ2pred_i < 0.00000000001
                λ2pred_i = 0.00000000001
            end

            if λ3pred_i < 0.00000000001
                λ3pred_i = 0.00000000001
            end

            logsurvival = sum(log.(survival) )
            if state[i] == 2
                loss += log(λ2pred_i) + logsurvival
            elseif state[i] == 3
                loss += log(λ3pred_i) + logsurvival
            elseif state[i] == 5
                loss += log(λ5pred_i) + logsurvival
            end
        #######################################################################################
        elseif (state[i] == 99) ## right censored
            timevec = collect(0:1:upper[i])' ./Tmax
            covs_i = hcat(fill(X[1:9,i], length(timevec))...)
            λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
            survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            if any(survival .< 0.00000000001)
                λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)) .* (1-λ2pred_i) #.* (1-λ2pred_i)
                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            end

            logsurvival = sum(log.(survival) )
            loss += logsurvival
        ###########################################################################################
        elseif (state[i] != 99) && (lower[i] != upper[i])## interval censored λ3 ->4
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()

            for jj in (lower_i+1):upper_i
                timevec = collect(0:1:jj)' ./Tmax
                covs_i = hcat(fill(X[1:9,i], length(timevec))...)

                λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
                if any(survival .< 0.00000000001)
                    λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                    λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)) .* (1-λ2pred_i)
                    survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
                end
                
    
                prodsurvival = prod(survival)

                if λ2pred_i < 0.00000000001
                    λ2pred_i = 0.00000000001
                end

                if λ3pred_i < 0.00000000001
                    λ3pred_i = 0.00000000001
                end

                if state[i] ==2
                    interval_prob += prodsurvival * λ2pred_i
                elseif state[i] == 3
                    interval_prob += prodsurvival * λ3pred_i
                elseif state[i] == 5
                    interval_prob += prodsurvival * λ5pred_i
                end
            end
            loss += log(interval_prob)
                    
    #######################################################################################
        end
    end

    return -1*loss , 1
end

function prediction_df(COVS::Matrix, TIME::Matrix)
    T2 = TIME[:,2]
    Nid = size(COVS)[2]
    patient_matrix_2 = DataFrame(zeros(maximum(T2)+1, Nid), :auto)
    patient_matrix_3 = DataFrame(zeros(maximum(T2)+1, Nid), :auto)
    patient_matrix_5 = DataFrame(zeros(maximum(T2)+1, Nid), :auto)
    max_time = maximum(T2)    
    time_in_state = collect(0:1:max_time)' ./ max_time

    for i in 1:Nid
        individual_covs = hcat(fill(COVS[1:9,i], length(time_in_state))...)
        individual_input_time = vcat(individual_covs, time_in_state)
        ipred2 = sigmoid.( vec(NN_model(individual_covs, ps.ps_λ2, ls2)[1] ) )
        ipred3 = sigmoid.( vec(NN_model(individual_covs, ps.ps_λ3, ls3)[1] ) )
        ipred5 = sigmoid.( vec(NN_model_time(individual_input_time, ps.ps_λ5, ls5)[1] ) )
        patient_matrix_2[!,i] = ipred2
        patient_matrix_3[!,i] = ipred3
        patient_matrix_5[!,i] = ipred5
    end

    return patient_matrix_2, patient_matrix_3, patient_matrix_5
end
#####################################################################################################
######## PARAMETER PRUNING ###########################
####################################################################################################
### To use the hessian or diaghessian functions from Zygote.jl,
## the parameters need to be in a vector. Therefore, use a 
## special loss function which take the "flat" vector of parameters
## and converts them into a parameter object used in a Lux model
## before calculating the loss

function loss1_parameter_pruning(params, params_prune, Network_pruned, ls2, ls3, ls5, state, X, T,  model23, model5, re=re)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]

    Tmax = maximum(upper)
    params_prune = re(params_prune)

    if Network_pruned == 2
        λ2pred = sigmoid(model23(X[1:9,:], params_prune, ls2)[1])
        λ3pred = sigmoid(model23(X[1:9,:], params.ps_λ3, ls3)[1])
        λ5pred = sigmoid(model5(X, params.ps_λ5, ls5)[1])
    elseif Network_pruned == 3
        λ2pred = sigmoid(model23(X[1:9,:], params.ps_λ2, ls2)[1])
        λ3pred = sigmoid(model23(X[1:9,:], params_prune, ls3)[1])
        λ5pred = sigmoid(model5(X, params.ps_λ5, ls5)[1])
    elseif Network_pruned == 5
        λ2pred = sigmoid(model23(X[1:9,:], params.ps_λ2, ls2)[1])
        λ3pred = sigmoid(model23(X[1:9,:], params.ps_λ3, ls3)[1])
        λ5pred = sigmoid(model5(X, params_prune, ls5)[1])
    end

    # λ2pred = sigmoid(model23(X[1:9,:], params.ps_λ2, ls2)[1])
    # λ3pred = sigmoid(model23(X[1:9,:], params.ps_λ3, ls3)[1])
    # λ5pred = sigmoid(model5(X, params.ps_λ5, ls5)[1])

    for i in eachindex(lower)
        λ2pred_i = λ2pred[i]
        λ3pred_i = λ3pred[i]
        λ5pred_i = λ5pred[i]
        if (λ2pred_i + λ3pred_i + λ5pred_i) > 0.999
            λ2pred_i = λ2pred_i * (1-λ5pred_i)
            λ3pred_i = λ3pred_i * (1-λ5pred_i-λ2pred_i)
        end

        #########################################################################################    
        if (state[i] != 99 ) && (lower[i] == upper[i]) ## we know the exact time of event 5
            timevec = collect(0:1:upper[i])' ./Tmax
            covs_i = hcat(fill(X[1:9,i], length(timevec))...)
            if Network_pruned == 5
                λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params_prune, ls5)[1] )
            elseif Network_pruned !=5
                λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
            end
            survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            if any(survival .< 0.00000000001)
                λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)-λ2pred_i) #.* (1-λ2pred_i)
                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            end

            if λ2pred_i < 0.00000000001
                λ2pred_i = 0.00000000001
            end

            if λ3pred_i < 0.00000000001
                λ3pred_i = 0.00000000001
            end

            logsurvival = sum(log.(survival) )

            lossi = 0.0
            if state[i] == 2
                lossi = log(λ2pred_i) + logsurvival
            elseif state[i] == 3
                lossi = log(λ3pred_i) + logsurvival
            elseif state[i] == 5
                lossi = log(λ5pred_i) + logsurvival
            end

            loss += lossi
        #######################################################################################
        elseif (state[i] == 99) ## right censored
            timevec = collect(0:1:upper[i])' ./Tmax
            covs_i = hcat(fill(X[1:9,i], length(timevec))...)
            if Network_pruned == 5
                λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params_prune, ls5)[1] )
            elseif Network_pruned !=5
                λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
            end
            survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            if any(survival .< 0.00000000001)
                λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)-λ2pred_i) #.* (1-λ2pred_i)
                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
            end

            logsurvival = sum(log.(survival) )
            loss += logsurvival
        ###########################################################################################
        elseif (state[i] != 99) && (lower[i] != upper[i])## interval censored λ3 ->4
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()


            for jj in (lower_i+1):upper_i
                timevec = collect(0:1:jj)' ./Tmax
                covs_i = hcat(fill(X[1:9,i], length(timevec))...)

                if Network_pruned == 5
                    λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params_prune, ls5)[1] )
                elseif Network_pruned !=5
                    λ5_time_i = sigmoid.(model5(vcat(covs_i, timevec), params.ps_λ5, ls5)[1] )
                end

                survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
                if any(survival .< 0.00000000001)
                    λ2pred_i = λ2pred_i .* (1-maximum(λ5_time_i))
                    λ3pred_i = λ3pred_i .* (1-maximum(λ5_time_i)-λ2pred_i) #.* (1-λ2pred_i)
                    survival = 1 .- (λ2pred_i .+ λ3pred_i .+ λ5_time_i)
                end
                
    
                prodsurvival = prod(survival)

                if λ2pred_i < 0.00000000001
                    λ2pred_i = 0.00000000001
                end

                if λ3pred_i < 0.00000000001
                    λ3pred_i = 0.00000000001
                end

                if state[i] ==2
                    interval_prob += prodsurvival * λ2pred_i
                elseif state[i] == 3
                    interval_prob += prodsurvival * λ3pred_i
                elseif state[i] == 5
                    interval_prob += prodsurvival * λ5pred_i
                end
            end
            lossi = log(interval_prob)

            loss += lossi
    #######################################################################################
        end
    end

    return -1*loss , 1
end
## Calculate salience of model parameters: θ^2 * diagonal element of hessian
function salience_parameters(params, params_prune, Network_pruned, ls2, ls3, ls5, state, COVS, TIME, model4, model5)
    Nid = size(TIME)[1]
    flat, re = destructure(params_prune)
    diagonalhess = diaghessian(p -> loss1_parameter_pruning(params, p, Network_pruned, ls2, ls3, ls5, state, COVS, TIME', model4, model5, re)[1], flat)[1]
    param_salience = (flat.^2) .* abs.(diagonalhess)
    return param_salience ./ Nid
end

## function to find the minimum non-zero element of vector
## required for identifying minimum non-zero elements of the salience vector used for
## parameter pruning
## returns the index of that element
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


function zero_params(params, params_prune, Network_pruned, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, parameters_to_remove)
    ## set lowest salience parameter to 0.0
    param_salience = salience_sampling(params, params_prune, Network_pruned, ls2, ls3, ls5, state, COVS, TIME,
                                        model23, model5, 10, 30, loss_fn::Function)

    
    max_sal = maximum(param_salience)
    if Network_pruned == 2
        flatls, re_ls = destructure(ls2)
    elseif Network_pruned == 3
        flatls, re_ls = destructure(ls3)
    elseif Network_pruned == 5
        flatls, re_ls = destructure(ls5)
    end

    Index_vector = Int32[]
    for i in 1:parameters_to_remove
        index = custom_argmin(param_salience)
        param_salience[index] = 0.0

        if Network_pruned != 5
            flatls[index] = 0.0

        elseif Network_pruned ==5
            flatls[index] = 0.0

            check9 = check_connection(9, COVS, flatls, re_ls)
            check10 = check_connection(10, COVS, flatls, re_ls)

            if check9 == true || check10 == true
                flatls[index] = 1.0
                param_salience[index] = max_sal ## to ensure parameter stays when removing disconnected parameters
                println("reinstating parameter $index.")
            end
            counter = 0.0
            while check9 == true || check10 == true
                if check9 == true
                    println("Age is removed, testing other parameter.")
                end

                if check10 == true
                    println("Time is removed, testing other parameter.")
                end
                counter += 1.0

                println("While loop iteration No $counter.")
                index = custom_argmin(param_salience)
                param_salience[index] = 0.0
                flatls[index] = 0.0

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
                end
                
                if check9 == false && check10 == false
                    push!(Index_vector, index)
                    println("AGE and TIME are connected!")
                    break
                end
            end
            #####################################################
            ## check if connection still holds for covs 1, 9 and 10
            #####################################################
        end
    end

    for i in eachindex(param_salience)
        if param_salience[i] == 0.0
            flatls[i] = 0.0
        end
    end

    layer_states = re_ls(flatls)

    return layer_states, Index_vector#, param_salience
end

function count_parameters(ps)
    flatps,_ = destructure(ps)
    return count(flatps .!=0.0)    
end

function network_pruning(ps, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function,
                        state_test, COVS_test, TIME_test,
                        number_of_final_parameters, Number_of_pruning_iterations)
    pruning_losses_train = []
    pruning_losses_test = []
    pruning_N_parameters2 = []
    pruning_N_parameters3 = []
    pruning_N_parameters5 = []
    pruning_parameter_index2 = []
    pruning_parameter_index3 = []
    pruning_parameter_index5 = []
    push!(pruning_losses_train, loss1_fn(ps, ls2, ls3, ls5, state, COVS, TIME', model23, model5)[1])
    push!(pruning_losses_test, loss1_fn(ps, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23, model5)[1])
    push!(pruning_N_parameters2, count_parameters(ls2))
    push!(pruning_N_parameters3, count_parameters(ls3))
    push!(pruning_N_parameters5, count_parameters(ls5))
    push!(pruning_parameter_index2, [0])
    push!(pruning_parameter_index3, [0])
    push!(pruning_parameter_index5, [0])

    for iteration_no in 1:Number_of_pruning_iterations
        if iteration_no == 1
            ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 20)
            ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 20)
            ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 20)
        elseif iteration_no ==2 || iteration_no == 3
            ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
            ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
            ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
        elseif iteration_no ==4 || iteration_no == 5
            ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
            ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
            ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 10)
        elseif iteration_no == 5 || iteration_no == 6
            ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 5)
            ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 5)
            ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 5)
        elseif iteration_no == 7 || iteration_no == 8
            ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 2)
            ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 2)
            ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 3)
        elseif iteration_no > 8
            parameters_left2 = count_parameters(ls2)
            parameters_left3 = count_parameters(ls3)
            parameters_left5 = count_parameters(ls5)
            if (parameters_left2 > number_of_final_parameters)
                ls2, Index_vector2 = zero_params(ps, ps.ps_λ2, 2, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 1)
            else
                Index_vector2 = [0]
            end

            if (parameters_left3 > number_of_final_parameters)
                ls3, Index_vector3 = zero_params(ps, ps.ps_λ3, 3, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 1)
            else
                Index_vector3 = [0]
            end

            if (parameters_left5 > number_of_final_parameters)
                ls5, Index_vector5 = zero_params(ps, ps.ps_λ5, 5, ls2, ls3, ls5, state, COVS, TIME, model23, model5, loss_fn::Function, 1)
            else
                Index_vector5 = [0]
            end

        end

        parameters_left2 = count_parameters(ls2)
        parameters_left3 = count_parameters(ls3)
        parameters_left5 = count_parameters(ls5)
        println("Starting training...")
        ps ,_ ,_ = training_test_train(1000, 25, loss1_fn, NN_model, NN_model_time, state, ps, ls2, ls3, ls5,
                                    COVS, TIME,
                                    state_test, COVS_test, TIME_test)

        
    
        train_loss = loss1_fn(ps, ls2, ls3, ls5, state, COVS, TIME', model23, model5)[1]
        test_loss = loss1_fn(ps, ls2, ls3, ls5, state_test, COVS_test, TIME_test', model23, model5)[1]
        push!(pruning_losses_train, train_loss)
        push!(pruning_losses_test, test_loss)
        push!(pruning_N_parameters2, parameters_left2)
        push!(pruning_N_parameters3, parameters_left3)
        push!(pruning_N_parameters5, parameters_left5)
        push!(pruning_parameter_index2, Index_vector2)
        push!(pruning_parameter_index3, Index_vector3)
        push!(pruning_parameter_index5, Index_vector5)


        @save "Results/SNNmodels/mod_2params$parameters_left2-3params$parameters_left3-5params$parameters_left5.jld2" ps ls2 ls3 ls5

    end
    pruning_df = DataFrame(loss_train = pruning_losses_train,
                        loss_test = pruning_losses_test,
                        No_parameters2 = pruning_N_parameters2,
                        No_parameters3 = pruning_N_parameters3,
                        No_parameters5 = pruning_N_parameters5,
                        parameter_index2 = pruning_parameter_index2,
                        parameter_index3 = pruning_parameter_index3,
                        parameter_index5 = pruning_parameter_index5)


    CSV.write("Results/SNNmodels/summary_pruning.csv", pruning_df)
    return ls2, ls3, ls5, pruning_df
end

function custom_sampler(X::Matrix, ids::Vector, stratify_by::Vector;
    n_per_category::Int,
    TIME::Union{Nothing, Matrix}=nothing,
    individual_list::Union{Nothing, AbstractVector}=nothing,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    # Input checks
    ncols = size(X, 2)
    @assert length(ids) == ncols "Length of ids must match number of columns in X"
    @assert length(ids) == length(stratify_by) "Length of stratify_by must match number of columns in X"

    # Map from individual => column indices
    individual_to_cols = Dict{eltype(ids), Vector{Int}}()
    for (i, id) in enumerate(ids)
        push!(get!(individual_to_cols, id, Int[]), i)
    end

    # Map from individual => category
    individual_to_cat = Dict(id => stratify_by[first(cols)] for (id, cols) in individual_to_cols)

    # Group individuals by category
    cat_to_inds = Dict{eltype(stratify_by), Vector{eltype(ids)}}()
    for (ind, cat) in individual_to_cat
        push!(get!(cat_to_inds, cat, eltype(ids)[]), ind)
    end

    # Sample individuals per category
    sampled_individuals = eltype(ids)[]
    sampled_individual_categories = eltype(stratify_by)[]
    for (cat, inds) in cat_to_inds
        @assert length(inds) ≥ n_per_category "Not enough individuals in category '$cat' to sample $n_per_category"
        sampled = sample(rng, inds, n_per_category; replace=false)
        append!(sampled_individuals, sampled)
        append!(sampled_individual_categories, fill(cat, n_per_category))
    end

    # For each sampled individual, collect their column indices, and expand ids per column
    sampled_cols = Int[]
    sampled_ids = eltype(ids)[]
    for ind in sampled_individuals
        cols = individual_to_cols[ind]
        append!(sampled_cols, cols)
        append!(sampled_ids, fill(ind, length(cols)))
    end

    # TIME matrix rows corresponding to sampled individuals (once per individual)
    TIME_sampled = nothing
    if TIME !== nothing
        @assert individual_list !== nothing "Must provide `individual_list` if `TIME` is given"
        id_to_row = Dict(id => i for (i, id) in enumerate(individual_list))
        row_indices = [id_to_row[id] for id in sampled_individuals]
        TIME_sampled = TIME[row_indices, :]
    end

    return X[:, sampled_cols], sampled_cols, sampled_ids, sampled_individuals, sampled_individual_categories, TIME_sampled
end

function salience_sampling(params, params_prune, Network_pruned, ls2, ls3, ls5, state, COVS, TIME,
                            model23, model5, samplings, Nid_per_sample, loss_fn::Function)
    Nid = size(TIME)[1]
    Ncovs = size(COVS)[1]
    ID = collect(1:1:Nid)
    n_params = length(destructure(params_prune)[1])
    dH = zeros(n_params, samplings)

    t = time()
    individual_list = 1:Nid
    for i in 1:samplings
        println("------------------------")
        println("Runing sample no $i...")
        td = (time() - t) / 60
        println("Time elapsed: $td min.")
        println("------------------------")
        ## sample at random from the covs and corresponding TIME 
        COVS_sampled, _, _,_, state_sampled, TIME_sampled = custom_sampler(COVS, ID, state,
                                                                      n_per_category=Nid_per_sample,
                                                                      TIME=TIME,
                                                                      individual_list=individual_list)

        dH[:, i] = salience_parameters(params, params_prune, Network_pruned, ls2, ls3, ls5, state_sampled, COVS_sampled, TIME_sampled, NN_model, NN_model_time)
        #println(dH)
    end

    medianhess = zeros(n_params)

    for jj in 1:n_params
        medianhess[jj] = median(dH[jj,:])
    end

    #runtime = time() -t
    return medianhess#, runtime
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
                push!(l_str, "($(input[i]))/(abs($(input[i+1])) + 1)")
                #push!(l_str, "($(input[i]))/($(input[i+1]) + 1)")
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
