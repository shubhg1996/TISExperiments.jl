using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using Crux, Flux, BSON, StaticArrays, Random
using MCTS
using FileIO
using Plots
unicodeplots()

# Basic MDP
tprob = 0.6
Random.seed!(0)
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=tprob)

# Learn a policy that solves it
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right])
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp)
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g))
# BSON.@save "demo/gridworld_policy_table.bson" atable

atable = BSON.load("examples/gridworld_policy_table.bson")[:atable]

# Define the adversarial mdp
adv_rewards = deepcopy(randcosts)
for (k,v) in mdp.g.rewards
    if v < 0
        adv_rewards[k] += -10*v
    end
end

amdp = GridWorldMDP(rewards=adv_rewards, tprob=1., discount=1., terminate_from=mdp.g.terminate_from)

# Define action probability for the adv_mdp
action_probability(mdp, s, a) = (a == atable[s][1]) ? tprob : ((1. - tprob) / (length(actions(mdp)) - 1.))


# Generic Discrete NonParametric with symbol support
struct GenericDiscreteNonParametric
    g_support::Any
    pm::DiscreteNonParametric
end

GenericDiscreteNonParametric(vs::T, ps::Ps) where {
        T<:Any,P<:Real,Ps<:AbstractVector{P}} = GenericDiscreteNonParametric([v for v in vs], DiscreteNonParametric([i for i=1:length(vs)], ps))

Distributions.support(d::GenericDiscreteNonParametric) = d.g_support

Distributions.probs(d::GenericDiscreteNonParametric)  = d.pm.p

function Base.rand(rng::AbstractRNG, d::GenericDiscreteNonParametric)
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    return x[i]
end

function Distributions.pdf(d::GenericDiscreteNonParametric, x::Any)
    s = support(d)
    idx = findfirst(isequal(x), s)
    ps = probs(d)
    if idx <= length(ps) && s[idx] == x
        return ps[idx]
    else
        return zero(eltype(ps))
    end
end
Distributions.logpdf(d::GenericDiscreteNonParametric, x::Any) = log(pdf(d, x))

function disturbance(m::typeof(amdp), s)
    xs = POMDPs.actions(m, s)
    ps = [action_probability(m, s, x) for x in xs]
    ps ./= sum(ps)
    px = GenericDiscreteNonParametric(xs, ps)
    return px
end

fixed_s = rand(initialstate(amdp))

import TISExperiments

N = 100_000
c = 0.3
α = 0.0001

β = 0.1
γ = 0.4

baseline = true

print(N, c, α, β, γ)

path = "data/gridworld"

tree_mdp = TISExperiments.construct_tree_amdp(amdp, disturbance; reduction="sum")

results_baseline, results_tis, planner = TISExperiments.run_baseline_and_treeIS(amdp, tree_mdp, fixed_s, disturbance; N=N, c=c, α=α, β=β, γ=γ, baseline=baseline)

if baseline
    print("Baseline metrics")

    TISExperiments.evaluate_metrics(results_baseline[1]; alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
end

print("\nTIS metrics: N=$(N), c=$(c), α=$(α), β=$(β)), γ=$(γ)")

TISExperiments.evaluate_metrics(results_tis[1]; weights=exp.(results_tis[3]), alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

if baseline
    save("$(path)_baseline_$(N).jld2", Dict("risks:" => results_baseline[1], "states:" => results_baseline[2]))
end

save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => results_tis[1], "states:" => results_tis[2], "IS_weights:" => results_tis[3], "tree:" => results_tis[4]))
