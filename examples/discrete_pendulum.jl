using Pkg
Pkg.activate("/home/users/shubhgup/Codes/Julia/TISExperiments.jl")
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using Crux, Flux, BSON
include("utils.jl")
include("discrete_exploration.jl")
using FileIO

## Setup and solve the mdp
# dt = 0.1
# mdp = InvertedPendulumMDP(λcost=1, Rstep=.1, dt=dt, px=Normal(0f0, 0.1f0))
# policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh), x-> x .* 2f0), 1), [0f0]),
#                      ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1))))
# policy = solve(PPO(π=policy, S=state_space(mdp), N=100000, ΔN=400, max_steps=400), mdp)
# BSON.@save "policies/pendulum_policy.bson" policy

policy = BSON.load("policies/pendulum_policy.bson")[:policy]
# Crux.gif(mdp, policy, "out.gif", max_steps=100,)


## Construct the risk estimation mdp where actions are disturbances
dt = 0.1 # Do not change
maxT = 2.0 # Do not change
px_nom = Normal(0f0, 0.35f0) # Do not change
# px_nom = Normal(0f0, 1.0f0)

# cost environment
env = InvertedPendulumMDP(dt=dt, failure_thresh=Inf)
costfn(m, s, sp) = isterminal(m, sp) ? abs(sp[2]) : 0
amdp = RMDP(env, policy, costfn, true, dt, maxT)

function disturbance(m::typeof(amdp), s)
    xs = [-1, 0, 1] # Do not change
    pxs = pdf.(px_nom, xs)
    pxs = pxs ./ sum(pxs)
    px = DiscreteNonParametric(xs, pxs)
    return px
end

fixed_s = rand(initialstate(amdp))

import TISExperiments

N = 100_000
c = 0.0
α = 0.01

β = 0.25
γ = 0.25
# β = 0.00
# γ = 0.00

schedule = 0.1

baseline = true

path = "data/discretependulum"

print("Starting grid search...")

# mc_samps = BSON.load("data/10mil_mcsamps.bson")[:mc_samps]
# mc_weights = ones(length(mc_samps))

# TISExperiments.run_grid_search(amdp, tree_mdp, fixed_s, disturbance, mc_samps, ones(length(mc_samps)), path)
# # TISExperiments.run_grid_search(amdp, tree_mdp, fixed_s, disturbance, mc_samps, ones(length(mc_samps)), path; N_l=[1_000_000], save_every=1)
# # TISExperiments.run_grid_search(amdp, tree_mdp, fixed_s, disturbance, mc_samps, ones(length(mc_samps)), path; N_l=[10_000], α_l=[1e-2], β_a_l=[0.5], β_b_l=[0.3], schedule_l = [0.1])

# print("...Completed.")

results_baseline, results_tis, planner = TISExperiments.run_baseline_and_treeIS(amdp, tree_mdp, fixed_s, disturbance; N=N, c=c, α=α, β=β, γ=γ, schedule=schedule, baseline=baseline);

if baseline
    print("Baseline metrics")

    TISExperiments.evaluate_metrics(results_baseline[1]; alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
else
    print("\nTIS metrics: N=$(N), c=$(c), α=$(α), β=$(β)), γ=$(γ)")

    TISExperiments.evaluate_metrics(results_tis[1]; weights=exp.(results_tis[3]), alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
end

if baseline
    save("$(path)_baseline_$(N).jld2", Dict("risks:" => results_baseline[1], "states:" => results_baseline[2]))
else
    save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => results_tis[1], "states:" => results_tis[2], "IS_weights:" => results_tis[3], "tree:" => results_tis[4]))
end
