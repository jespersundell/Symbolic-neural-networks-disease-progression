#############################################################
####### SIMULATION FUNCTIONS FOR WHOLE DATA SET #############
#############################################################

function create_covariate_dataset(Nid)

    ## base on distribution from validation dataset
    d_debutalder = truncated(Normal(61.0, 11.95), 18, 95)

    ID = collect(1:1:Nid)
    sex = rand([0.5, -0.5], Nid)
    trig = rand(0:0.001:1, Nid)
    bmi = rand(0:0.001:1, Nid)
    hba1c = rand(0:0.001:1, Nid)
    sys = rand(0:0.001:1, Nid)
    dia = rand(0:0.001:1, Nid)
    hdl = rand(0:0.001:1, Nid)
    ldl = rand(0:0.001:1, Nid)
    debutalder = Int.(round.(rand(d_debutalder, Nid)))

    cov_df = DataFrame(ID = ID, x1 = sex,
                        x2 = trig,
                        x3 = bmi,
                        x4 = hba1c,
                        x5 = sys,
                        x6 = dia,
                        x7 = hdl,
                        x8 = ldl,
                        age_mid=debutalder)

    return cov_df
end

function sim_state_1(COVS::Vector, max_time::Int)

    time_vector = Int[]
    time_vector_current_state = Int[]
    state = Int[]
    T_total = 0
    T_in_state = 0
    max_time_state = 114

    T_in_state_normal = T_in_state/max_time_state

    ################### for age(t) ##########################
    max_age = 102.07
    age_years = COVS[9] + T_total/12 ## age in age_years
    age_cov = age_years / max_age
    ##########################
    current_state = 1

    λ12f(AGE) = sigmoid( -7.35 + 3.34*AGE^2 )
    λ13f(AGE) = sigmoid( -3.77 + -4.41(1.61*AGE)^(-2.26*AGE) ) 
    λ15f(AGE, T) = sigmoid( -5.1 + 2.93*T + 3.96*AGE - 3.67AGE^(-1.32*AGE))
    λ12_time = λ12f(age_cov)
    λ13_time = λ13f(age_cov)
    λ15_time = λ15f(age_cov, (T_in_state/max_time_state))

    λ11_time = 1-(λ12_time+λ13_time+λ15_time)
    d_event = Categorical(λ11_time, λ12_time, λ13_time, λ15_time)
    event = rand(d_event)
    if event != 1
        if event == 2
            push!(state, 12)
            current_state = 2
        elseif event == 3
            push!(state, 13)
            current_state = 3
        elseif event == 4
            push!(state, 15)
            current_state = 5 
        end
        push!(time_vector, T_in_state)
        push!(time_vector_current_state, T_in_state)
    end

    while (event == 1)
        T_total += 1
        T_in_state += 1
        T_in_state_normal = T_in_state/max_time_state
        ################### for age(t) ##########################
        age_years = COVS[9] + T_total/12 ## age in age_years
        age_cov = age_years / max_age
        ##########################
        λ12_time = λ12f(age_cov)
        λ13_time = λ13f(age_cov)
        λ15_time = λ15f(age_cov, (T_in_state/max_time_state))

        λ11_time = 1-(λ12_time+λ13_time+λ15_time)
        d_event = Categorical(λ11_time, λ12_time, λ13_time, λ15_time)
        event = rand(d_event)
        if event != 1
            if event == 2
                push!(state, 12)
                current_state = 2
            elseif event == 3
                push!(state, 13)
                current_state = 3
            elseif event == 4
                push!(state, 15)
                current_state = 5 
            end
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            break
        end

        if T_total == max_time -1 ## -1 since starts at 0
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            push!(state, 199)
            break
        elseif T_in_state == max_time_state -1
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            push!(state, 199)
            break
        end

    end

    return time_vector, state, T_total, current_state, time_vector_current_state
end

function sim_state_2(COVS::Vector, max_time, T_total, time_vector, state, time_vector_current_state)

    T_in_state = 0
    current_state = 2
    max_time_state = 95
    T_in_state_normal = T_in_state/max_time_state

    λ24f(AGE) =  sigmoid( -13.69 + 9.08*AGE )
    λ25f(AGE, T) = sigmoid( -11.64 + 5.87*AGE + 5.56*T*AGE )
    ################### for age(t) ##########################
    max_age = 102.07
    age_years = COVS[9] + T_total/12 ## age in age_years
    age_cov = age_years / max_age
    ##########################

    λ24_time = λ24f(age_cov)
    λ25_time = λ25f(age_cov, (T_in_state/max_time_state))

    λ22_time = 1-(λ24_time+λ25_time)

    d_event = Categorical(λ22_time, λ24_time, λ25_time)
    event = rand(d_event)
    if event != 1
        if event == 2
            push!(state, 24)
            current_state = 4
        elseif event == 3
            push!(state, 25)
            current_state = 5
        end
        push!(time_vector, T_total)
        push!(time_vector_current_state, T_in_state)
    end

    while (event == 1)
        T_total += 1
        T_in_state += 1
        T_in_state_normal = T_in_state/max_time_state
        ################### for age(t) ##########################
        age_years = COVS[9] + T_total/12 ## age in age_years
        age_cov = age_years / max_age
        ##########################
        λ24_time = λ24f(age_cov)
        λ25_time = λ25f(age_cov, (T_in_state/max_time_state))
        λ22_time = 1-(λ24_time+λ25_time)

        d_event = Categorical(λ22_time, λ24_time, λ25_time)
        event = rand(d_event)

        if event != 1
            if event == 2
                push!(state, 24)
                current_state = 4
            elseif event == 3
                push!(state, 25)
                current_state = 5
            end
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            break
        end

        if T_total == max_time -1 ## -1 since starts at 0
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            push!(state, 299)
            break
        elseif T_in_state == max_time_state -1
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            push!(state, 299)
            break
        end

    end

    return time_vector, state, T_total, current_state, time_vector_current_state
end

function sim_state_3(COVS::Vector, max_time, T_total, time_vector, state, time_vector_current_state)

    T_in_state = 0
    current_state = 3
    max_time_state = 97

    T_in_state_normal = T_in_state/max_time_state
    λ34f(AGE) = sigmoid( -5.88 )
    λ35f(AGE, T) = sigmoid( -4.33 / ((0.34T + 0.79AGE)^1.98 + 0.33) )
    ################### for age(t) ##########################
    max_age = 102.07
    age_years = COVS[9] + T_total/12 ## age in age_years
    age_cov = age_years / max_age
    ##########################
    λ34_time = λ34f(age_cov)
    λ35_time = λ35f(age_cov, (T_in_state/max_time_state) )

    λ33_time = 1-(λ34_time+λ35_time)

    d_event = Categorical(λ33_time, λ34_time, λ35_time)
    event = rand(d_event)
    if event != 1
        if event == 2
            push!(state, 34)
            current_state = 4
        elseif event == 3
            push!(state, 35)
            current_state = 5
        end
        push!(time_vector, T_total)
        push!(time_vector_current_state, T_in_state)
    end

    while (event == 1)
        T_total += 1
        T_in_state += 1
        T_in_state_normal = T_in_state/max_time_state
        ################### for age(t) ##########################
        age_years = COVS[9] + T_total/12 ## age in age_years
        age_cov = age_years / max_age
        ##########################

        λ34_time = λ34f(age_cov)
        λ35_time = λ35f(age_cov, (T_in_state/max_time_state) )
        λ33_time = 1-(λ34_time+λ35_time)

        d_event = Categorical(λ33_time, λ34_time, λ35_time)
        event = rand(d_event)

        if event != 1
            if event == 2
                push!(state, 34)
                current_state = 4
            elseif event == 3
                push!(state, 35)
                current_state = 5
            end
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            break
        end

        if T_total == max_time -1 ## -1 since starts at 0
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            push!(state, 399)
            break
        elseif T_in_state == max_time_state -1
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            push!(state, 399)
            break
        end

    end

    return time_vector, state, T_total, current_state, time_vector_current_state
end

function sim_state_4(COVS::Vector, max_time, T_total, time_vector::Vector, state::Vector, time_vector_current_state)

    T_in_state = 0
    current_state = 4
    max_time_state = 78
    max_age = 102.07
    
    T_in_state_normal = T_in_state/max_time_state

    λ45f(AGE, T) = sigmoid( -12.8 + 4.28*T + 8.47*AGE )

    ################### for age(t) ##########################
    max_age = 102.07
    age_years = COVS[9] + T_total/12 ## age in age_years
    age_cov = age_years / max_age
    ##########################
    λ45_time = λ45f(age_cov, (T_in_state/max_time_state))
    λ44_time = 1-λ45_time

    d_event = Categorical(λ44_time, λ45_time)
    event = rand(d_event)
    if event != 1
        if event == 2
            push!(state, 45)
            current_state = 5
        end
        push!(time_vector, T_total)
        push!(time_vector_current_state, T_in_state)
    end

    while (event == 1)
        T_total += 1
        T_in_state += 1
        T_in_state_normal = T_in_state/max_time_state
        ################### for age(t) ##########################
        age_years = COVS[9] + T_total/12 ## age in age_years
        age_cov = age_years / max_age
        ##########################
        λ45_time = λ45f(age_cov, (T_in_state/max_time_state))
        λ44_time = 1-λ45_time

        d_event = Categorical(λ44_time, λ45_time)
        event = rand(d_event)

        if event != 1
            if event == 2
                push!(state, 45)
                current_state = 5
            end
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            break
        end

        if T_total == max_time -1 ## -1 since starts at 0
            push!(time_vector, T_total)
            push!(time_vector_current_state, T_in_state)
            push!(state, 499)
            break
        elseif T_in_state == max_time_state -1
            push!(time_vector, T_in_state)
            push!(time_vector_current_state, T_in_state)
            push!(state, 499)
            break
        end

    end
    return time_vector, state, T_total, current_state, time_vector_current_state
end

function simulator(COVS::Vector, max_time)
    time_vector, state, T_total, current_state, time_vector_current_state = sim_state_1(COVS, max_time)

    if (current_state == 2) && (T_total < max_time)
        time_vector, state, T_total, current_state, time_vector_current_state = sim_state_2(COVS, max_time, T_total, time_vector, state, time_vector_current_state)
    elseif (current_state == 3) && (T_total < max_time)
        time_vector, state, T_total, current_state, time_vector_current_state = sim_state_3(COVS, max_time, T_total, time_vector, state, time_vector_current_state)
    end

    if (current_state == 4) && (T_total < max_time)
        time_vector, state, T_total, current_state, time_vector_current_state = sim_state_4(COVS, max_time, T_total, time_vector, state, time_vector_current_state)
    end


    id_df = DataFrame(time = time_vector, state = state, time_current = time_vector_current_state)
    return id_df
end

function simulate!(COVS_mat::Matrix, max_time)
    Nid = size(COVS_mat)[2]

    sim_df = DataFrame(time = Int[], state = Int[], ID = Int[])

    for i in 1:Nid
        icovs = COVS_mat[:,i]
        id_df = simulator(icovs, max_time)
        id = fill(i, length(id_df.time))
        id_df.ID = id
        sim_df = vcat(sim_df, id_df)
    end

    return sim_df
end

function add_censoring(df, censoring_distribution::MixtureModel)
    Nid = length(unique(df.ID))
    Probability_of_censored_obs = (9799/11611) ## account for that not all observations are censored
    RC = Int.(round.(rand(censoring_distribution, Nid) .* (1/Probability_of_censored_obs) ) )
    grpID = groupby(df, :ID)

    for i in 1:Nid
        Nobs = length(grpID[i].ID)
        RCi = fill(RC[i], Nobs)
        grpID[i].RCt = RCi

        if grpID[i].RCt[end] >= grpID[i].time[end]
            grpID[i].diff = fill(99, Nobs)
            ## variable for filtering observations after censoring
            grpID[i].KEEP = fill(1, Nobs)
            continue
        elseif grpID[i].RCt[end] < grpID[i].time[end]
            ## logic for what happens if censoring time is lower than time of event
            grpID[i].diff = grpID[i].RCt .- grpID[i].time
            ind = findfirst(grpID[i].diff .<0)
            ## adjust time to  time of censoring
            grpID[i].time[ind] = grpID[i].RCt[ind] 
            ## adjust state to right censored transit at time  == ind
            if (grpID[i].state[ind] == 12) || (grpID[i].state[ind] == 13) || (grpID[i].state[ind] == 15) || (grpID[i].state[ind] == 199)
                grpID[i].state[ind] = 199
            elseif (grpID[i].state[ind] == 24) || (grpID[i].state[ind] == 25) || (grpID[i].state[ind] == 299)
                grpID[i].state[ind] = 299
            elseif (grpID[i].state[ind] == 34) || (grpID[i].state[ind] == 35) || (grpID[i].state[ind] == 399)
                grpID[i].state[ind] = 399
            elseif (grpID[i].state[ind] == 45) || (grpID[i].state[ind] == 499)
                grpID[i].state[ind] = 499
            end
            ## variable for filtering observations after censoring
            observations_to_keep = fill(1, ind)
            if Nobs > ind
                observations_to_remove = fill(0, Nobs-ind)
            else
                observations_to_remove = Int[]
            end
            grpID[i].KEEP = vcat(observations_to_keep, observations_to_remove)
        end

        if (length(grpID[i].time) > 1) && (grpID[i].time[end] == grpID[i].time[end-1])
            grpID[i].time[end] = grpID[i].time[end] +1
        end
    end

    ## remove observation after index == ind
    df = filter(row -> row.KEEP == 1, df)
    return df 
end

function simulate_censoring!(COVS_mat::Matrix, max_time, mixture_distribution::MixtureModel)
    Nid = size(COVS_mat)[2]

    sim_df = DataFrame(time = Int[], time_current = Int[], state = Int[], ID = Int[])

    for i in 1:Nid
        icovs = COVS_mat[:,i]
        id_df = simulator(icovs, max_time)
        id = fill(i, length(id_df.time))
        id_df.ID = id
        sim_df = vcat(sim_df, id_df)
    end

    sim_df = add_censoring(sim_df, mixture_distribution)
    return sim_df
end

function covariate_matrix(df::DataFrame)
    Nid = size(df)[1]
    mat = zeros(9, Nid)
    for i in 1:Nid
        mat[:,i] = collect(df[i, 2:end])
    end

    return mat
end

#############################################################################################
## simulates a dataset with all transitions and includes right censoring
function simulate_full_dataset(Nid, max_time)
    censoring_distribution = MixtureModel([Exponential(21.1), Gamma(9, 6.4)], [0.5, 1 - 0.5])
    df_covs = create_covariate_dataset(Nid)
    COVS = covariate_matrix(df_covs)
    df_sim = simulate_censoring!(COVS, max_time, censoring_distribution)
    df_sim = select(df_sim, Not(:diff, :RCt, :KEEP))

    df_sim.trans = df_sim.state
    df = outerjoin(df_sim, df_covs, on=:ID)
    #rename!(df, :time => :Tl)
    df.Tl = df.time

    df.state[df.state.==12] .= 2
    df.state[df.state.==13] .= 3
    df.state[df.state.==15] .= 5
    df.state[df.state.==199] .= 99

    df.state[df.state.==24] .= 4
    df.state[df.state.==25] .= 5
    df.state[df.state.==299] .= 99

    df.state[df.state.==34] .= 4
    df.state[df.state.==35] .= 5
    df.state[df.state.==399] .= 99

    df.state[df.state.==45] .= 5
    df.state[df.state.==499] .= 99

    grpID = groupby(df, :ID)

    for i in 1:Nid
        Nobs = size(grpID[i])[1]
        for j in 1:Nobs
            if j == 1
                grpID[i].time_current[j] = grpID[i].Tl[j]
            elseif j >1
                grpID[i].time_current[j] = grpID[i].Tl[j] - grpID[i].Tl[j-1]
            end
        end
    end 

    df.Tl = df.time_current
    df.Tu = df.Tl
    df = select(df, Not(:time_current))
    df = select(df, [:Tl, :Tu, :time, :state, :ID, :trans, :x1, :x2, :x3, :x4, :x5, :x6, :x7, :x8, :age_mid])
    return df
end

#####################################################################

function filter_data_state(df, state::Int)
    if state == 1
        filter_trans = [12, 13, 15, 199]
    elseif state == 2
        filter_trans = [24, 25, 299]
    elseif state == 3
        filter_trans = [34, 35, 399]
    elseif state == 4
        filter_trans = [45, 499]
    end

    filtered_df = filter(row -> (row.trans in filter_trans), df)

    return filtered_df
end

#############################################################
####### VPC related functions #############
#############################################################
function AJ_sim(COVS_mat, max_time, iterations, mixture_distribution::MixtureModel)
    sim_P1 = zeros(max_time, iterations)
    sim_P2 = zeros(max_time, iterations)
    sim_P3 = zeros(max_time, iterations)
    sim_P4 = zeros(max_time, iterations)
    sim_P5 = zeros(max_time, iterations)
    tsim = collect(0:1:max_time-1)

    for i in 1:iterations
        if i % 5 ==0
            println("Iteration number $i")
        end

        sim_df = simulate_censoring!(COVS_mat, max_time, mixture_distribution::MixtureModel)
        #sim_df = simulate!(COVS_mat, max_time)
        _, πt = Aalen_Johansen_estimator(sim_df, max_time)

        sim_P1[:,i] = πt[1,:] 
        sim_P2[:,i] = πt[2,:] 
        sim_P3[:,i] = πt[3,:]
        sim_P4[:,i] = πt[4,:] 
        sim_P5[:,i] = πt[5,:]

    end

    return tsim, sim_P1, sim_P2, sim_P3, sim_P4, sim_P5
end

function AJ_summary_stats(COVS_mat, max_time, iterations, mixture_distribution::MixtureModel)

    sim_time, P1, P2, P3, P4, P5  = AJ_sim(COVS_mat, max_time, iterations, mixture_distribution)
    summary_df = DataFrame( time = sim_time,
                            P1_med = zeros(max_time),
                            P1_max = zeros(max_time),
                            P1_min = zeros(max_time),

                            P2_med = zeros(max_time),
                            P2_max = zeros(max_time),
                            P2_min = zeros(max_time),

                            P3_med = zeros(max_time),
                            P3_max = zeros(max_time),
                            P3_min = zeros(max_time),

                            P4_med = zeros(max_time),
                            P4_max = zeros(max_time),
                            P4_min = zeros(max_time),

                            P5_med = zeros(max_time),
                            P5_max = zeros(max_time),
                            P5_min = zeros(max_time))


    for i in 1:max_time
        P1clean_vec = replace(P1[i,:], NaN => 0.0)
        summary_df.P1_med[i] = median(P1clean_vec)
        summary_df.P1_max[i] = maximum(P1clean_vec)
        summary_df.P1_min[i] = minimum(P1clean_vec)

        P2clean_vec = replace(P2[i,:], NaN => 0.0)
        summary_df.P2_med[i] = median(P2clean_vec)
        summary_df.P2_max[i] = maximum(P2clean_vec)
        summary_df.P2_min[i] = minimum(P2clean_vec)

        P3clean_vec = replace(P3[i,:], NaN => 0.0)
        summary_df.P3_med[i] = median(P3clean_vec)
        summary_df.P3_max[i] = maximum(P3clean_vec)
        summary_df.P3_min[i] = minimum(P3clean_vec)

        P4clean_vec = replace(P4[i,:], NaN => 0.0)
        summary_df.P4_med[i] = median(P4clean_vec)
        summary_df.P4_max[i] = maximum(P4clean_vec)
        summary_df.P4_min[i] = minimum(P4clean_vec)

        P5clean_vec = replace(P5[i,:], NaN => 0.0)
        summary_df.P5_med[i] = median(P5clean_vec)
        summary_df.P5_max[i] = maximum(P5clean_vec)
        summary_df.P5_min[i] = minimum(P5clean_vec)

    end

    return summary_df
end

function VPC_plot(obstime, π, sim_df, plottitle::String)
    p1 = plot(obstime, π[1,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability")
    plot!(obstime, sim_df.P1_med, title="State 1", color="red", label=false)
    plot!(obstime, sim_df.P1_max,
                fillrange= sim_df.P1_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    p2 = plot(obstime, π[2,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability")
    plot!(obstime, sim_df.P2_med, title="State 2", color="red", label=false)
    plot!(obstime, sim_df.P2_max,
                fillrange= sim_df.P2_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)
    

    p3 = plot(obstime, π[3,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability")
    plot!(obstime, sim_df.P3_med, title="State 3", color="red", label=false)
    plot!(obstime, sim_df.P3_max,
                fillrange= sim_df.P3_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    p4 = plot(obstime, π[4,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability")
    plot!(obstime, sim_df.P4_med, title="State 4", color="red", label=false)
    plot!(obstime, sim_df.P4_max,
                fillrange= sim_df.P4_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    p5 = plot(obstime, π[5,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability")
    plot!(obstime, sim_df.P5_med, title="State 5", color="red", label=false)
    plot!(obstime, sim_df.P5_max,
                fillrange= sim_df.P5_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    pcomb = plot(p1, p2, p3, p4, p5, plot_title = plottitle, size=(800, 750))
    return pcomb
end

function Nelson_Aalen(df, max_time)
    #N_ID = length(df.time)
    #N_ID = length(unique(df.ID))

    if "ID" in names(df)
        N_ID = length(unique(df.ID))
    else
        N_ID = length(df.time)
    end
    NA_time = zeros(max_time)

    at_risk1 = zeros(max_time)
    at_risk2 = zeros(max_time)
    at_risk3 = zeros(max_time)
    at_risk4 = zeros(max_time)

    N_events_12, N_events_13, N_events_15, N_events_1DO = zeros(max_time), zeros(max_time), zeros(max_time), zeros(max_time)
    N_events_24, N_events_25, N_events_2DO = zeros(max_time), zeros(max_time), zeros(max_time)
    N_events_34, N_events_35, N_events_3DO = zeros(max_time), zeros(max_time), zeros(max_time)    
    N_events_45, N_events_4DO = zeros(max_time), zeros(max_time)

    event_times_12 = df.time[df.state .== 12]
    event_times_13 = df.time[df.state .== 13]
    event_times_15 = df.time[df.state .== 15]
    event_times_1DO = df.time[df.state .== 199]

    event_times_24 = df.time[df.state .== 24]
    event_times_25 = df.time[df.state .== 25]
    event_times_2DO = df.time[df.state .== 299]

    event_times_34 = df.time[df.state .== 34]
    event_times_35 = df.time[df.state .== 35]
    event_times_3DO = df.time[df.state .== 399]

    event_times_45 = df.time[df.state .== 45]
    event_times_4DO = df.time[df.state .== 499]

    trans_intensity_12, trans_intensity_13, trans_intensity_15 = zeros(max_time), zeros(max_time), zeros(max_time)
    trans_intensity_24, trans_intensity_25 = zeros(max_time), zeros(max_time)
    trans_intensity_34, trans_intensity_35 = zeros(max_time), zeros(max_time)
    trans_intensity_45 = zeros(max_time)

    #N_events_14 = zeros(max_time)
    event_times_14 = df.time[df.state .== 14]
    trans_intensity_14 = zeros(max_time) 

    for i in 1:max_time
        NA_time[i] = i-1 ## stating at 0
        N_events_12[i] = count(event_times_12 .== i-1)
        N_events_13[i] = count(event_times_13 .== i-1)
        #N_events_14[i] = count(event_times_14 .== i-1)
        N_events_15[i] = count(event_times_15 .== i-1)
        N_events_1DO[i] = count(event_times_1DO .== i-1)

        N_events_24[i] = count(event_times_24 .== i-1)
        N_events_25[i] = count(event_times_25 .== i-1)
        N_events_2DO[i] = count(event_times_2DO .== i-1)

        N_events_34[i] = count(event_times_34 .== i-1)
        N_events_35[i] = count(event_times_35 .== i-1)
        N_events_3DO[i] = count(event_times_3DO .== i-1)

        N_events_45[i] = count(event_times_45 .== i-1)
        N_events_4DO[i] = count(event_times_4DO .== i-1)

        if i == 1
            at_risk1[i] = N_ID
            at_risk2[i] = 0
            at_risk3[i] = 0
            at_risk4[i] = 0                
            trans_intensity_12[i], trans_intensity_13[i], trans_intensity_15[i] = 0, 0, 0
            #trans_intensity_14[i] = 0
            trans_intensity_24[i], trans_intensity_25[i] = 0, 0
            trans_intensity_34[i], trans_intensity_35[i] = 0, 0
            trans_intensity_45[i] = 0


        elseif i > 1
            at_risk1[i] = at_risk1[i-1] - (N_events_12[i-1]+N_events_13[i-1]+N_events_15[i-1]+N_events_1DO[i-1]) #+N_events_14[i-1]
            trans_intensity_12[i] = N_events_12[i-1] / at_risk1[i-1]
            trans_intensity_13[i] = N_events_13[i-1] / at_risk1[i-1]
            #trans_intensity_14[i] = N_events_14[i-1] / at_risk1[i-1]
            trans_intensity_15[i] = N_events_15[i-1] / at_risk1[i-1]

            ## update at_risk with incoming from state 1
            at_risk2[i] = at_risk2[i-1] +N_events_12[i-1] - (N_events_24[i-1]+N_events_25[i-1]+N_events_2DO[i-1])
            if at_risk2[i-1] > 0
                trans_intensity_24[i] = N_events_24[i-1] / at_risk2[i-1]
                trans_intensity_25[i] = N_events_25[i-1] / at_risk2[i-1]
            else
                trans_intensity_24[i] = 0
                trans_intensity_25[i] = 0
            end

            at_risk3[i] = at_risk3[i-1] +N_events_13[i-1] - (N_events_34[i-1]+N_events_35[i-1]+N_events_3DO[i-1])
            if at_risk3[i-1] > 0
                trans_intensity_34[i] = N_events_34[i-1] / at_risk3[i-1]
                trans_intensity_35[i] = N_events_35[i-1] / at_risk3[i-1]
            else
                trans_intensity_34[i] = 0
                trans_intensity_35[i] = 0
            end

            at_risk4[i] = at_risk4[i-1] + ( N_events_24[i-1] + N_events_34[i-1]) - (N_events_45[i-1]+N_events_4DO[i-1]) # N_events_14[i-1] +
            if at_risk4[i-1] > 0
                trans_intensity_45[i] = N_events_45[i-1] / at_risk4[i-1]
            else
                trans_intensity_45[i] = 0
            end
        end
    end

    return NA_time, trans_intensity_12, trans_intensity_13, trans_intensity_15, #trans_intensity_14,
                    trans_intensity_24, trans_intensity_25,
                    trans_intensity_34, trans_intensity_35,
                    trans_intensity_45      
    #return N_events_25                  
end
#NA_time, ti12, ti13, ti14, ti15, ti24, ti25, ti34, ti35, ti45 = Nelson_Aalen(df, 100)
function Aalen_Johansen_estimator(df, max_time)
    π0 = [1, 0, 0, 0, 0]'
    discrete_time, ti12, ti13, ti15, ti24, ti25, ti34, ti35, ti45 = Nelson_Aalen(df, max_time)

    πt = zeros(length(π0), max_time)

    for i in 1:max_time
        ti11_i= (1- (ti12[i]+ti13[i]+ti15[i]) )
        ti22_i= (1- (ti24[i]+ti25[i]) )
        ti33_i= (1- (ti34[i]+ti35[i]) )
        ti44_i= (1- (ti45[i]) )

        T_i = [ti11_i ti12[i] ti13[i] 0 ti15[i];
               0 ti22_i 0   ti24[i] ti25[i];
               0  0 ti33_i  ti34[i] ti35[i];
               0  0 0  ti44_i ti45[i];
               0    0   0   0   1]
        #################
        if i == 1
            πt[:,i] = π0 * T_i
        elseif i > 0
            πt[:,i] = πt[:, i-1]' * T_i
        end

    end

    state4 = πt[4,:]
    for ii in eachindex(state4)
        if state4[ii] < 0
            state4[ii] = 0.0
        end
    end
    πt[4,:] = state4
    return discrete_time, πt
end

function AJ_plot(obstime, π, plottitle::String)
    p1 = plot(obstime, π[1,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability",
                title = "State 1")

    p2 = plot(obstime, π[2,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability",
                title = "State 2")
    
    p3 = plot(obstime, π[3,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability",
                title="State 3")

    p4 = plot(obstime, π[4,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability",
                title="State 4")
    p5 = plot(obstime, π[5,:], color="grey", label=false,
                xaxis="Time (months)",
                yaxis="Probability",
                title="State 5")

    pcomb = plot(p1, p2, p3, p4, p5, plot_title = plottitle, size=(800, 750))
    return pcomb
end

#############################################################
####### Bootstrap related functions #############
#############################################################
function bootstrap_subject_resample(df::DataFrame, id_col::Symbol = :ID)
    # Get unique subject IDs
    unique_ids = unique(df[:, id_col])
    
    # Number of subjects
    n = length(unique_ids)
    
    # Sample IDs with replacement
    sampled_ids = rand(1:n, n)  # indices for resampled subjects
    resampled_ids = unique_ids[sampled_ids]
    
    # Collect rows corresponding to sampled subject IDs,
    # replicating subjects if they appear multiple times
    resampled_rows = DataFrame[]
    ID_count = 0
    for id in resampled_ids
        ID_count += 1
        df_i = filter(row -> row[id_col] == id, df)
        nobs = length(df_i.ID)
        new_ID_col = fill(ID_count, nobs)
        df_i.ID = new_ID_col
        push!(resampled_rows, df_i)
    end
    
    # Concatenate all subject rows into one DataFrame for the bootstrap sample
    return vcat(resampled_rows...)
    #return resampled_rows
end

function bootstrap_raw_data(df, N_bootstraps)
    max_time= maximum(df.time)

    s1 = zeros(max_time, N_bootstraps)
    s2 = zeros(max_time, N_bootstraps)
    s3 = zeros(max_time, N_bootstraps)
    s4 = zeros(max_time, N_bootstraps)
    s5 = zeros(max_time, N_bootstraps)

    t = time()
    for i in 1:N_bootstraps
        df_boot = bootstrap_subject_resample(df)
        _, πt_boot = Aalen_Johansen_estimator(df_boot, max_time)
        s1[:, i] = πt_boot[1,:]
        s2[:, i] = πt_boot[2,:]
        s3[:, i] = πt_boot[3,:]
        s4[:, i] = πt_boot[4,:]
        s5[:, i] = πt_boot[5,:]

        dt = (time() -t) / 60
        println("Runing bootstrap iteration no $i.")
        println("Elapsed time is $dt min.")
        println("------------------------------")
    end

    s1[isnan.(s1)] .= 0
    s2[isnan.(s2)] .= 0
    s3[isnan.(s3)] .= 0
    s4[isnan.(s4)] .= 0
    s5[isnan.(s5)] .= 0

    s1df = DataFrame(s1, :auto)
    s2df = DataFrame(s2, :auto)
    s3df = DataFrame(s3, :auto)
    s4df = DataFrame(s4, :auto)
    s5df = DataFrame(s5, :auto)
    
    return s1df, s2df, s3df, s4df, s5df
end

function bootstrap_summary(df::DataFrame)
    timepoints = size(df)[1]

    med = zeros(timepoints)
    CI_low = zeros(timepoints)
    CI_upper = zeros(timepoints)
    for i in 1:timepoints
        bs_t = collect(df[i, :])
        med[i] = median(bs_t)
        CI_low[i] = quantile(bs_t, 0.025)
        CI_upper[i] = quantile(bs_t, 0.975)
    end
    return hcat(med, CI_low, CI_upper)
end

function VPC_bootstrap_plot(s1sum, s2sum, s3sum, s4sum, s5sum, sim_df, plottitle::String)
    plot_time = sim_df.time./12
    p1 = plot(plot_time, s1sum[:,1], color="grey", label=false,
                xaxis="Time (years)",
                yaxis="Probability")
    plot!(plot_time, s1sum[:,2],
                fillrange= s1sum[:,3],
                linestyle=:dash, color="grey",
                alpha= 0.3,
                label=false)
    plot!(plot_time, sim_df.P1_med, title="State 1", color="red", label=false)
    plot!(plot_time, sim_df.P1_max,
                fillrange= sim_df.P1_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)
        
    p2 = plot(plot_time, s2sum[:,1], color="grey", label=false,
                xaxis="Time (years)",
                yaxis="Probability")
        plot!(plot_time, s2sum[:,2],
                fillrange= s2sum[:,3],
                linestyle=:dash, color="grey",
                alpha= 0.3,
                label=false)
    plot!(plot_time, sim_df.P2_med, title="State 2", color="red", label=false)
    plot!(plot_time, sim_df.P2_max,
                fillrange= sim_df.P2_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)
    

    p3 = plot(plot_time, s3sum[:,1], color="grey", label=false,
                xaxis="Time (years)",
                yaxis="Probability")
        plot!(plot_time, s3sum[:,2],
                fillrange= s3sum[:,3],
                linestyle=:dash, color="grey",
                alpha= 0.3,
                label=false)
    plot!(plot_time, sim_df.P3_med, title="State 3", color="red", label=false)
    plot!(plot_time, sim_df.P3_max,
                fillrange= sim_df.P3_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    p4 = plot(plot_time, s4sum[:,1], color="grey", label=false,
                xaxis="Time (years)",
                yaxis="Probability")
        plot!(plot_time, s4sum[:,2],
                fillrange= s4sum[:,3],
                linestyle=:dash, color="grey",
                alpha= 0.3,
                label=false)
    plot!(plot_time, sim_df.P4_med, title="State 4", color="red", label=false)
    plot!(plot_time, sim_df.P4_max,
                fillrange= sim_df.P4_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    p5 = plot(plot_time, s5sum[:,1], color="grey", label=false,
                xaxis="Time (years)",
                yaxis="Probability")
        plot!(plot_time, s5sum[:,2],
                fillrange= s5sum[:,3],
                linestyle=:dash, color="grey",
                alpha= 0.3,
                label=false)
    plot!(plot_time, sim_df.P5_med, title="State 5", color="red", label=false)
    plot!(plot_time, sim_df.P5_max,
                fillrange= sim_df.P5_min,
                linestyle=:dash, color="red",
                alpha= 0.3,
                label=false)

    pcomb = plot(p1, p2, p3, p4, p5, plot_title = plottitle, size=(800, 750))
    return pcomb
end

#############################################################
####### Pseudo observations and state occupation #############
#############################################################
function pseudo_observations(df, πtobs::Vector, time_point)
    ID_no = unique(df.ID)
    Nid = length(ID_no)
    pseudo_obs = zeros(Nid, 5)
    #testvec = []
    for i in eachindex(ID_no)
        if i % 1000 == 0
            println("Runing individual number $i.")
        end

        ID = ID_no[i]
        df_not_i = filter(row -> row.ID != ID, df)
        _, πtobs_hat = Aalen_Johansen_estimator(df_not_i, time_point)
        #push!(testvec, πtobs_hat)
        πtobs_not_i = πtobs_hat[:,time_point]
        pseudo_obs[i,:] = (Nid .* πtobs) .- ((Nid-1) .* πtobs_not_i )
    end
    return pseudo_obs    
end

function normalize_pseudo_observations(po::Matrix)
    Nid = length(po[:,1])
    po_normalized = zeros(Nid, 5)
    for i in 1:Nid
        ipo = po[i,:]
        for jj in eachindex(ipo)
            if ipo[jj] < 0.0
                ipo[jj] = 0.00000000001
            end
        end

        po_normalized[i,:] = ipo ./ sum(ipo)
    end
    return po_normalized
end

function state_occupancy(df, time_point)
    grpID = groupby(df, :ID)
    Nid = length(grpID)
    state_occupancy=zeros(Nid, 5)
    id_vec = zeros(Nid)
    for i in 1:Nid
        id_vec[i] = grpID[i].ID[1]
        Nobs = length(grpID[i].ID)
        if Nobs == 1             ## no transitions to consider
            if grpID[i].time[1] > time_point
                state_occupancy[i,1] = 1.0
            elseif grpID[i].time[1] <= time_point
                if grpID[i].state[1] == 15 ## death from state 1
                    state_occupancy[i,5] = 1.0
                elseif grpID[i].state[1] == 199 ## rigth censored from state 1
                    state_occupancy[i,:] .= 99
                end
            end
        elseif Nobs > 1 && grpID[i].time[end] <=time_point
                if (grpID[i].state[end] == 25) || (grpID[i].state[end] == 35) || (grpID[i].state[end] == 45) ## death from state other than 1
                    state_occupancy[i,5] = 1.0
                elseif (grpID[i].state[end] == 299) || (grpID[i].state[end] == 399) || (grpID[i].state[end] == 499)## rigth censored from state other than 1
                    state_occupancy[i,:] .= 99
                end
        elseif Nobs > 1 && grpID[i].time[end] > time_point## transitions to consider
            time_index = findfirst(>(time_point), grpID[i].time)
            if time_index == 1
                state_occupancy[i,1] = 1.0
            else
                println(i)
                current_state = grpID[i].start[time_index]
                state_occupancy[i,current_state] = 1.0

            end    

        end

    end

    x = 99
    rows_to_keep = .!([x in row for row in eachrow(state_occupancy)])
    state_occupancy_filtered = state_occupancy[rows_to_keep, :]
    id_vec_filtered = id_vec[rows_to_keep]
    return state_occupancy_filtered, id_vec_filtered
end

function firstdigit(x::Integer)
    iszero(x) && return x
    x = abs(x)
    y = 10^floor(Int, log10(x))
    return div(x, y)
end


#############################################################
####### Model Predictions Kunina #############
#############################################################
function prediction_matrix(covs, end_time)
    Nid = size(covs)[1]
    ipred_t = zeros(Nid, 5)

    u0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    tspan = (0.0, end_time)
    for i in 1:Nid
        icovs = covs[i,:]
        prob = ODEProblem(kunina_model, u0, tspan, icovs)
        sol = solve(prob)
        ipred_t[i,:] = sol.u[end]
    end
    return ipred_t
end

function kunina_model(du, u, icovs, T)
    a1, a2, a3, a4 = u
    SEX, BTRI, BBMI, BHBA1C, BSYS, BDIAS, BHDL, BLDL, DEBUT = icovs ## replace with cov 

    SLOPE = 3.44
    MtAS25 = 5.0
    MtAS35 = 9.68
    MtAS45 = 12.2
    #1/MTT24 = 0.016 ## change
    #1/MTT34 = 0.066 ## change
    β12 = 0.00125
    κ13 = 0.089
    β13 = 14e-06

    if SEX == 0 ## female
        κ15 = 0.086
        α15 = 0.00051
        β15 = 13.4e-06
        κ12 = 0.047
    elseif SEX == 1 ## male
        κ15 = 0.082
        α15 = 0.00115
        β15 = 24.9e-06
        κ12 = 0.0524
    end      
                                            
    ## Gompertz Makeham formula
    T2DP = SLOPE * T
    k15 = α15 + β15 * exp(κ15*(DEBUT+T+T2DP))
    k25 = α15 + β15 * exp(κ15*(DEBUT+T+MtAS25+T2DP))
    k35 = α15 + β15 * exp(κ15*(DEBUT+T+MtAS35+T2DP))
    k45 = α15 + β15 * exp(κ15*(DEBUT+T+MtAS45+T2DP))    

    k12 = β12 * exp(κ12*(DEBUT+T))
    k13 = β13 * exp(κ13*(DEBUT+T))
    k24 = 0.016
    k34 = 0.066

    du[1] = -k13*a1 - k12*a1 - k15*a1
    du[2] =  k12*a1 - k25*a2 - k24*a2 
    du[3] =  k13*a1 - k35*a3 - k34*a3
    du[4] =  k24*a2 + k34*a3 - k45*a4
    du[5] =  k15*a1 + k25*a2 + k35*a3 + k45*a4
end

function back_transform_covariates(covariate_matrix)

    covs = deepcopy(covariate_matrix)
    sex = covs[:,1]
    for i in eachindex(sex)
        if sex[i] == 0.5
            sex[i] = 1
        elseif sex[i] == -0.5
            sex[i] = 0
        end
    end

    covs[:,1] = sex
    covs[:,2] = covs[:,2] .*16
    covs[:,3] = covs[:,3] .*50
    covs[:,4] = covs[:,4] .*144
    covs[:,5] = covs[:,5] .*250
    covs[:,6] = covs[:,6] .*130
    covs[:,7] = covs[:,7] .*4
    covs[:,8] = covs[:,8] .*8.65

    return covs
    
end

#############################################################
####### Model Performance #############
#############################################################
function covariate_matrix_pred(df::DataFrame)
    Nid = size(df)[1]
    mat = zeros(Nid, 9)
    for i in 1:Nid
        mat[i,:] = collect(df[i, 2:end])
    end

    return mat
end

function KL_divergence_i(P::Vector, Q::Vector)
    ## to avoid 0 probabilities 
    for i in eachindex(P)
        if P[i] == 0.0
            P[i] = eps()
        end
    end
    D_KL = P .* log.(P ./ Q)
    return sum(D_KL)
end

function KL_divergence(df_obs::DataFrame, df_pred::DataFrame)
    Nid = size(df_obs)[1]

    KL_vector = zeros(Nid)

    for i in 1:Nid
        ipred = collect(df_pred[i,:])
        iobs = collect(df_obs[i,:])
        KL_vector[i] = KL_divergence_i(iobs, ipred)
    end

    return KL_vector
end

function Brier_score_i(obs::Vector, pred::Vector)
    BS = (obs .- pred).^2
    return sum(BS)
end

function Brier_score(df_obs::DataFrame, df_pred::DataFrame)
    Nid = size(df_obs)[1]

    BS_vector = zeros(Nid)

    for i in 1:Nid
        ipred = collect(df_pred[i,:])
        iobs = collect(df_obs[i,:])
        BS_vector[i] = Brier_score_i(iobs, ipred)
    end

    return BS_vector
end


#############################################################
####### Model Predictions SNN #############
#############################################################
function individual_predicted_prob(icovs::Vector, end_time)
    π0 = [1, 0, 0, 0, 0]'
    #discrete_time = collect(0:1:end_time)

    πt = zeros(length(π0), end_time+1)
    πt[1] = 1.0

    AGE = icovs[9] ## age in years
    max_age = 102.07

    for i in 1:end_time
        T = i-1  ## time in months
        AGE_at_T = AGE + (T /12) ## age in years
        AGE_normalized = AGE_at_T / max_age
        T1 = T/114
        T2 = T/95
        T3 = T/97
        T4 = T/78

        λ12_i = λ12(icovs, AGE_normalized)
        λ13_i = λ13(icovs, AGE_normalized)
        λ24_i = λ24(icovs, AGE_normalized)
        λ34_i = λ34(icovs, AGE_normalized)

        λ15_i = λ15(icovs, T1, AGE_normalized)
        λ25_i = λ15(icovs, T2, AGE_normalized)
        λ35_i = λ15(icovs, T3, AGE_normalized)
        λ45_i = λ15(icovs, T4, AGE_normalized)

        λ11_i= (1- (λ12_i+λ13_i+λ15_i) )
        λ22_i= (1- (λ24_i+λ25_i) )
        λ33_i= (1- (λ34_i+λ35_i) )
        λ44_i= (1- (λ45_i) )

        T_i = [λ11_i λ12_i λ13_i 0 λ15_i;
               0 λ22_i 0   λ24_i λ25_i;
               0  0 λ33_i  λ34_i λ35_i;
               0  0 0  λ44_i λ45_i;
               0    0   0   0   1]
        #################
        πt[:,i+1] = πt[:, i]' * T_i
    end
    return πt[:,end]
end

function prediction_matrix_SNN(covs::Matrix, end_time)
    Nid = size(covs)[1]
    ipred_t = zeros(Nid, 5)

    for i in 1:Nid
        ipred_t[i,:] = individual_predicted_prob(covs[i,:], end_time)
    end
    return ipred_t
end