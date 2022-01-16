using POMDPs, POMDPGym
using Distributions
# using FileIO
using Random

struct SimpleMDP <: MDP{Float64, Float64} end

POMDPs.initialstate(mdp::SimpleMDP) = MersenneTwister()

POMDPs.actions(mdp::SimpleMDP) = DiscreteNonParametric([1.0, 2.0], [0.5, 0.5])
POMDPs.isterminal(mdp::SimpleMDP, s) = s > 5
POMDPs.discount(mdp::SimpleMDP) = 1.0

function POMDPs.gen(mdp::SimpleMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    sp = s + a
    r = sp > 2 ? 1 : 2
    return (sp=sp, r=r)
end

mdp = SimpleMDP()

fixed_s = rand(initialstate(mdp))

import TISExperiments

N = 1000
c = 0.3
α = 0.1

path = "/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/simple"

tree_mdp = TISExperiments.construct_tree_amdp(mdp, actions; reduction="sum")

results_baseline, results_tis, planner = TISExperiments.run_baseline_and_treeIS(mdp, tree_mdp, fixed_s, actions; N, c, α)

# save("$(path)_baseline_$(N).jld2", Dict("risks:" => results_baseline[1], "states:" => results_baseline[2]))

# save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => results_tis[1], "states:" => results_tis[2], "IS_weights:" => results_tis[3], "tree:" => results_tis[4]))
