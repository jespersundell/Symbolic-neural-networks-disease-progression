####################
## Mixture model to estimate right censoring distribution
###################

using DataFrames, CSV
using Statistics, Distributions, Random
using Plots
using ExpectationMaximization
###########################################################################
include("s00_simulator_functions.jl")
###########################################################################
### Load datasets ########
###########################################################################
df = CSV.read("Data/Full_dataset.csv", DataFrame)
###########################################################################

###########################################################################
### Estimate probability of rigth censoring using a mixture model ########
###########################################################################
censored_obs_types = [199, 299, 399, 499]
df_censored = filter(row -> (row.trans in censored_obs_types), df)
# Your observed positive data
data = (df_censored.Tl .+1)
# Create mixture model based on initial parameters
mix_guess = MixtureModel([Exponential(0.1), Gamma(0.5, 1)], [0.5, 1 - 0.5])
# Fit mixture model to data using EM algorithm from ExpectationMaximization.jl
mixture_fit = fit_mle(mix_guess, data; display = :iter, atol = 1e-3, robust = false, infos = false)

# max_time = maximum(df_censored.Tu)
# xs = range(0, stop=max_time, length=100)  
# ys = pdf.(mixture_fit, xs)     


# pmix = histogram(df_censored.Tu, normalize=:pdf, alpha = 0.2, label=false)
# plot!(xs, ys, xlabel="Censoring times (months)",
#               ylabel="Density",
#               label = false,
#               lw=3,
#               tickfontsize = 14,
#               guidefontsize = 16)

#savefig(pmix, "Results/censoring_times_pdf.png")

mixture_fit.components