using Pkg
Pkg.activate("/home/users/shubhgup/Codes/Julia/TISExperiments.jl")
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using Crux, Flux, BSON, StaticArrays, Random
using MCTS
using FileIO
# using Plots
# unicodeplots()

# Basic MDP
tprob = 0.7
Random.seed!(0)
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
# zerocosts = Dict(POMDPGym.GWPos(i,j) => 0.0 for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=tprob)

# Learn a policy that solves it
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right])
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp)
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g))
# BSON.@save "demo/gridworld_policy_table.bson" atable

atable = BSON.load("examples/gridworld_policy_table.bson")[:atable]

# Define the adversarial mdp
adv_rewards = deepcopy(randcosts)
# adv_rewards = deepcopy(zerocosts)
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

import TreeImportanceSampling, TISExperiments

N = 10_000
# N = 100_00_000
c = 0.0
α = 1e-4

β = 0.00
γ = 0.15

schedule = 1e-1 # set to Inf to switch off

uniform_floor = 0.00001 # set to v.small to switch off

alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

baseline = false
do_save = false

path = "data/gridworld_viz"

function run_baseline()
    println("Running Baseline N=$(N)")

    baseline_costs = []
    baseline_states = []
    for i in 1:N
        sim = simulate(HistoryRecorder(), amdp, FunctionPolicy((s) -> rand(disturbance(amdp, s))), fixed_s)
        push!(baseline_costs, sum(collect(sim[:r])))
        push!(baseline_states, collect(sim[:s]))
    end

    println("Baseline metrics")

    TISExperiments.evaluate_metrics(baseline_costs; alpha_list)

    if do_save
        save("$(path)_baseline_$(N).jld2", Dict("risks:" => Float64.(baseline_costs), "states:" => baseline_states))
    end

    return Float64.(baseline_costs), baseline_states
end

function run_tis()
    println("Running TIS N=$(N), c=$(c), α=$(α), β=$(β)), γ=$(γ)")

    tree_mdp = TreeImportanceSampling.TreeMDP(amdp, 1.0, [], [], disturbance, "sum")

    planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)

    a, info = action_info(planner, TreeImportanceSampling.TreeState(fixed_s); tree_in_info=true, β, γ, schedule, uniform_floor)

    println("TIS metrics")

    TISExperiments.evaluate_metrics(planner.mdp.costs; weights=exp.(planner.mdp.IS_weights), alpha_list)

    if do_save
        save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => Float64.(planner.mdp.costs), "states:" => [], "IS_weights:" => Float64.(planner.mdp.IS_weights), "tree:" => info[:tree]))
    end

    return Float64.(planner.mdp.costs), Float64.(planner.mdp.IS_weights), info[:tree], planner
end


# if baseline
#     save("$(path)_baseline_$(N).jld2", Dict("risks:" => results_baseline[1], "states:" => results_baseline[2]))
# else
#     save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => results_tis[1], "states:" => results_tis[2], "IS_weights:" => results_tis[3], "tree:" => results_tis[4]))
# end
