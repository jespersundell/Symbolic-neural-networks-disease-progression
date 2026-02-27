
###########################################################
####### Functions ########################################
#########################################################
function covariate_transformation_age_change(TIME, COVS, state)
    upper = TIME[:,2]
    time_vector = []
    id_vec = []
    state_vec = []
    max_time = maximum(upper)

    ## New for age chaning as inputs
    upper_months = upper ./12
    #max_age = maximum(upper_months .+ COVS[9,:])
    max_age = 102.07

    for i in eachindex(upper)
        timevec = collect(0:1:upper[i])' ./max_time ## added normalization here
        covs_i = hcat(fill(COVS[1:end-1,i], length(timevec))...)
        timevec_months = collect(0:1:upper[i]) ./12 ## New for age chaning as inputs
        covs_i[9,:] = (covs_i[9,:] .+ timevec_months) ./ max_age ## New for age chaning as inputs

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

    return COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test
    #COVS ./ maximum(df.age_mid), COVS_test./ maximum(df_test.age_mid)
end

function init_ps()
    NN_model = SNNmodel(10)
    ps_λ5, ls5 = Lux.setup(rng, NN_model)
    ps = ComponentArray{Float32}()
    ps = ComponentArray(ps;ps_λ5)
    loss = loss4(ps, ls5, state, COVS_new, TIME', NN_model, ID_new)[1]

    println("Training loss is $loss.")
    while isinf(loss) || loss > 1500
        NN_model = SNNmodel(10)
        ps_λ5, ls5 = Lux.setup(rng, NN_model)
        ps = ComponentArray{Float32}()
        ps = ComponentArray(ps;ps_λ5)
        loss = loss4(ps, ls5, state, COVS_new, TIME', NN_model, ID_new)[1]
        println("Training loss is $loss.")

        if !isinf(loss) && loss < 1500
            println("Training loss is $loss.")
            break
        end
    end
    
    return ps
end

function rank_mean_hessian(mean_hessian)
    p = sortperm(mean_hessian)          # Indices that sort v
    ranks = similar(p)       # Prepare a vector for ranks
    ranks[p] = 1:length(mean_hessian)   # Assign ranks based on sorted order

    return ranks
end

function prepare_data_for_NN(df, df_test)
    ################################################################################################
    #################################### TRAINING DATA ################################################
    covs = zeros(8, length(df.state))
    covdf = df[:,6:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    TIME = hcat(df.Tl, df.Tu)
    state = df.state
    Nid = size(TIME)[1]
    AGE = df.age_mid ./ maximum(df.age_mid)
    COVS = hcat(AGE...)
    COVS = vcat(covs, COVS)
    T2 = hcat(zeros(Nid)...)
    COVS = vcat(COVS, T2)
    #ID = collect(1:1:Nid)
    #COVS_new, ID_new, _ = covariate_transformation(TIME, COVS, state)
    ################################################################################################
    #################################### TEST DATA ################################################
    covs = zeros(8, length(df_test.state))
    covdf = df_test[:,6:end-1]
    for i in 1:8
        covs[i,:] = covdf[:,i]
    end
    TIME_test = hcat(df_test.Tl, df_test.Tu)
    state_test = df_test.state
    Nid_test = size(TIME_test)[1]
    AGE = df_test.age_mid ./ maximum(df_test.age_mid)
    COVS_test = hcat(AGE...)
    COVS_test = vcat(covs, COVS_test)
    T2 = hcat(zeros(Nid_test)...)
    COVS_test = vcat(COVS_test, T2)
    #ID_test = collect(1:1:Nid_test)
    #COVS_new_test, ID_new_test, _ = covariate_transformation(TIME_test, COVS_test, state_test)

    #return COVS_new, TIME, state, ID_new, COVS_new_test, TIME_test, state_test, ID_new_test, COVS, COVS_test
    return COVS, TIME, state, COVS_test, TIME_test, state_test
end

function loss4(params, ls, state, X, T,  model, idvec)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]

    λ5pred = sigmoid(model(X, params.ps_λ5, ls)[1])

    for i in eachindex(lower)
        λ5pred_i = λ5pred[idvec.==i]
        #########################################################################################    
        if (state[i] == 5 ) && (lower[i] == upper[i]) ## we know the exact time of event 5
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            λ5pred_i_jump = λ5pred_i[end]
            loss += log(λ5pred_i_jump) + survival
        #######################################################################################
        elseif (state[i] == 99) ## right censored λ4
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            loss += survival
        ###########################################################################################
        elseif (state[i] == 5) && (lower[i] != upper[i])## interval censored λ4 ->5
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()
            for jj in (lower_i+1):upper_i
                λ5_time_i = λ5pred_i[1:jj]
                survival = prod(1 .- λ5_time_i)
                λ5pred_i_jump = λ5pred_i[end]
                interval_prob += survival * λ5pred_i_jump
            end
            loss += log(interval_prob)
    #######################################################################################
        end
    end

    return -1*loss , 1
end

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
#####################################################################################################
######## PARAMETER PRUNING ###########################
####################################################################################################
### To use the hessian or diaghessian functions from Zygote.jl,
## the parameters need to be in a vector. Therefore, use a 
## special loss function which take the "flat" vector of parameters
## and converts them into a parameter object used in a Lux model
## before calculating the loss
function loss4_parameter_pruning(params, ls, state, X, T,  model, idvec, re=re)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]

    params = re(params)
    λ5pred = sigmoid(model(X, params.ps_λ5, ls)[1])

    for i in eachindex(lower)
        λ5pred_i = λ5pred[idvec.==i]
        #########################################################################################    
        if (state[i] == 5 ) && (lower[i] == upper[i]) ## we know the exact time of event 5
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            λ5pred_i_jump = λ5pred_i[end]
            loss += log(λ5pred_i_jump) + survival
        #######################################################################################
        elseif (state[i] == 99) ## right censored λ4
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            loss += survival
        ###########################################################################################
        elseif (state[i] == 5) && (lower[i] != upper[i])## interval censored λ4 ->5
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()
            for jj in (lower_i+1):upper_i
                λ5_time_i = λ5pred_i[1:jj]
                survival = prod(1 .- λ5_time_i)
                λ5pred_i_jump = λ5pred_i[end]
                interval_prob += survival * λ5pred_i_jump
            end
            loss += log(interval_prob)
    #######################################################################################
        end
    end

    return -1*loss , 1
end

function training_test_train(epochs, patience, lossfn::Function, model, state, parameters, layerstates, COVS, TIME, ID,
                            state_test, COVS_test, TIME_test, ID_test
                            ; opt_state = opt_state)
    lossvec = []
    lossvec_test = []
    push!(lossvec,lossfn(parameters, layerstates, state, COVS, TIME', model, ID)[1] )
    push!(lossvec_test,lossfn(parameters, layerstates, state_test, COVS_test, TIME_test', model, ID_test)[1] )
    t = time()

    ## for early stopping
    #patience = 10 # Number of epochs to wait before stopping
    min_delta = 0.0001 # Minimum change in loss to qualify as an improvement
    patience_counter = 0
    best_loss = Inf
    for epoch in 1:epochs
        (loss, _), back = pullback(lossfn, parameters, layerstates, state, COVS, TIME',  model, ID)# ## updated to train_loader (x=COVS, y=TIME)
        grad, _ = back((one(loss), nothing))

        opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
        losscount=loss

        if epoch % 50 == 0
            losscount = lossfn(parameters, layerstates, state, COVS, TIME', model, ID)[1]
            losscount_test = lossfn(parameters, layerstates, state_test, COVS_test, TIME_test', model, ID_test)[1]
            dt = time() - t
            println("Epoch: $epoch, Loss Train: $losscount")
            println("Epoch: $epoch, Loss Test: $losscount_test")
            println("Time elapsed: $dt sec")
            println("-------------------------------------")
        end
        if epoch % 1 == 0
            loss_train = lossfn(parameters, layerstates, state, COVS, TIME', model, ID)[1]
            loss_test = lossfn(parameters, layerstates, state_test, COVS_test, TIME_test', model, ID_test)[1]
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
        if isnan(lossvec[end])
            break
        end
    end
    return parameters, lossvec, lossvec_test   
end

## Calculate salience of model parameters: θ^2 * diagonal element of hessian
function salience_parameters(params, ls, state, COVS, TIME, model, ID)
    Nid = size(TIME)[1]
    flat, re = destructure(params)
    diagonalhess = diaghessian(p -> loss4_parameter_pruning(p, ls, state, COVS, TIME', model, ID, re)[1], flat)[1]
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
        output1 = NN_model(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model(testcov2, ps.ps_λ5, ls5)[1][1]
    elseif covnumber == 9
        id1_check[9] = 10
        testcov = hcat(vcat(id1, [0]) )
        testcov2 = hcat(vcat(id1_check, [0]) )
        output1 = NN_model(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model(testcov2, ps.ps_λ5, ls5)[1][1]
    elseif covnumber == 1
        id1_check[1] = 10
        testcov = hcat(vcat(id1, [0]) )
        testcov2 = hcat(vcat(id1_check, [0]) )
        output1 = NN_model(testcov, ps.ps_λ5, ls5)[1][1]
        output2 = NN_model(testcov2, ps.ps_λ5, ls5)[1][1]
    end
    
    return output1 ≈ output2
end

function zero_params(params, ls, state, COVS, TIME, model, ID, parameters_to_remove)
    ## set lowest salience parameter to 0.0
    param_salience = salience_parameters(params, ls, state, COVS, TIME, model, ID)
    max_sal = maximum(param_salience)

    flatls, re_ls = destructure(ls)
    Index_vector = Int32[]
    for i in 1:parameters_to_remove
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
#parameters, layerstates, index_sal = zero_params(parameters, layerstates)

##############################################################################
## Hessian-based pruning of covariates
#############################################################################
##############################################################################
function diaghessian_inputs(params, ls, COVS, TIME, model)
    diaghessvec = Float64[]
    N = size(COVS)[2]
    dH = diaghessian(x -> loss_fn(params, ls, x, TIME', model)[1], COVS)[1]
    for row in eachrow(dH)
        jointhess = sum(abs.(row)) * (1/N) ## normalized by number of individuals
        push!(diaghessvec, jointhess)
    end

    return diaghessvec    
end
#true_hess = diaghessian_inputs(NN_parameters, NN_ls_λ12, COVS, TIME, NN_model_λ12)

using StatsBase
## smapler with replacement which takes into account the distribution of a label which is state in this case
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

function loss4_covariate_pruning(params, ls, state, X, T,  model, idvec)# X = covariates, T = time to event
    loss = 0.0f0
    lower = T[1,:]
    upper = T[2,:]

    λ5pred = sigmoid(model(X, params.ps_λ5, ls)[1])

    for i in eachindex(lower)
        #λ5pred_i = λ5pred[idvec.==i]
        λ5pred_i = λ5pred[idvec.==unique(idvec)[i]]
        #########################################################################################    
        if (state[i] == 5 ) && (lower[i] == upper[i]) ## we know the exact time of event 5
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            λ5pred_i_jump = λ5pred_i[end]
            loss += log(λ5pred_i_jump) + survival
        #######################################################################################
        elseif (state[i] == 99) ## right censored λ4
            survival = sum(log.(1 .- λ5pred_i[1:end-1]))
            loss += survival
        ###########################################################################################
        elseif (state[i] == 5) && (lower[i] != upper[i])## interval censored λ4 ->5
            upper_i = upper[i]
            lower_i = lower[i]
            interval_prob = eps()
            for jj in (lower_i+1):upper_i
                λ5_time_i = λ5pred_i[1:jj]
                survival = prod(1 .- λ5_time_i)
                λ5pred_i_jump = λ5pred_i[end]
                interval_prob += survival * λ5pred_i_jump
            end
            loss += log(interval_prob)
    #######################################################################################
        end
    end

    return -1*loss , 1
end

function diaghessian_inputs_sampling(params, ls, COVS, TIME, model, state, samplings, Nid_per_sample, ID, loss_fn::Function)
    Nid = size(TIME)[1]
    Ncovs = size(COVS)[1]
    diaghessmatrix = zeros(samplings,Ncovs)

    t = time()
    individual_list = 1:Nid
    for j in 1:samplings
        println("------------------------")
        println("Runing sample no $j...")
        td = (time() - t) / 60
        println("Time elapsed: $td min.")
        println("------------------------")
        ## sample at random from the covs and corresponding TIME 
        COVS_sampled, _, ID_sampled,_, state_sampled, TIME_sampled = custom_sampler(COVS, ID, state,
                                                                      n_per_category=Nid_per_sample,
                                                                      TIME=TIME,
                                                                      individual_list=individual_list)

        dH = abs.(diaghessian(x -> loss_fn(params, ls, state_sampled, x, TIME_sampled', model, ID_sampled)[1], COVS_sampled)[1] )
        for (i, row) in enumerate(eachrow(dH))
            jointhess = sum(abs.(row)) * (1/(Nid_per_sample)) ## normalized by no ID per sample
            diaghessmatrix[j,i] = jointhess
        end
    end

    meanhess = zeros(Ncovs)

    for i in 1:Ncovs
        meanhess[i] = sum(diaghessmatrix[:,i]) *(1/samplings)
    end

    return meanhess    
end

#diaghessian_inputs_sampling(ps, ls5, COVS_new, TIME, NN_model, state_new, 20, 10, ID_new, loss4_covariate_pruning)

function remove_covariate(params, ls, COVS, TIME, model, state, samplings, Nid_per_sample, ID, loss_fn::Function)

    mean_hessian = diaghessian_inputs_sampling(params, ls, COVS, TIME, model, state, samplings, Nid_per_sample, ID, loss_fn::Function)
    removable_covariates = mean_hessian[1:8]
    minimum_contributing_covariate_index = custom_argmin(removable_covariates)
    #params_old = deepcopy(params)
    ls_old = deepcopy(ls)
    #params.ps_λ5.layer_1.weight[:,minimum_contributing_covariate_index] .= 0.0
    ls.layer_1.W_mask[:,minimum_contributing_covariate_index] .= 0.0

    covariate_names = ["Sex", "Triglycerider", "BMI", "hba1c", "Sys", "Dia", "hdl", "ldl"]
    for i in eachindex(removable_covariates)
        println("Covariate: ", covariate_names[i], ". Sensitivity: ", round(removable_covariates[i], digits=3) )
    end
    
    
    println("#############################################################")
    println("Removed covariate no $minimum_contributing_covariate_index: ", covariate_names[minimum_contributing_covariate_index],".")
    println("#############################################################")

    return params, ls, ls_old, mean_hessian
end
#################################################################################
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
                push!(l_str, "($(input[i]))/($(input[i+1]) + 1)")
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
#################################################################################
##############################################################################
## MODEL FUNCTIONS
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
