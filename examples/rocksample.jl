using Pkg
Pkg.activate("/home/users/shubhgup/Codes/Julia/TISExperiments.jl")
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using BSON
using SARSOP, Random
using RockSample
using BeliefUpdaters
using FileIO

Random.seed!(42)
pomdp = RockSamplePOMDP(rocks_positions=[(2,3), (4,4), (4,2), (5,3), (6,2)],
                        sensor_efficiency=10.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0)

randcosts = Dict(RockSample.RSPos(i,j) => 10.0 + 3*rand() for i = 1:10, j=1:10)
randcosts_a = Dict(Float64(i) => 3*rand() for i = 1:50)

solver = SARSOPSolver(precision=1e-3)
# policy = solve(solver, pomdp)
policy = load_policy(pomdp, "policy.out")

dt = 0.1 # Do not change
maxT = 2.0 # Do not change

function costfn(x)
    st, a, sp, o, r = x
    randcosts[st[2].pos] - r + randcosts_a[a]
end
amdp = RPOMDP(pomdp=pomdp, π=policy, updater=DiscreteUpdater(pomdp), initial_belief_distribution = uniform_belief(pomdp), cost_fn=costfn, dt=dt, maxT=maxT)

function action_probability(mdp, s, x)
    d = POMDPs.observation(mdp.pomdp, action(mdp.π, s[3]), s[2])
    d.probs[findfirst(d.vals .== x)]
end

function disturbance(m::typeof(amdp), s)
    xs = POMDPs.actions(m, s)
    ps = [action_probability(m, s, x) for x in xs]
    ps ./= sum(ps)
    px = DiscreteNonParametric(xs, ps)
    return px
end

fixed_s = rand(initialstate(amdp))

import TISExperiments

N = 10_000
c = 0.0
α = 0.01

# β = 0.25
# γ = 0.25
β = 0.00
γ = 0.00

schedule = 0.1

baseline = true

path = "data/rocksample"

tree_mdp = TISExperiments.construct_tree_amdp(amdp, disturbance; reduction="sum")

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
