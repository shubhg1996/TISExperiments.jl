using POMDPs, POMDPGym
using Distributions
using FileIO
using Random

struct SimpleMDP <: MDP{Float64, Float64} end

POMDPs.initialstate(mdp::SimpleMDP) = MersenneTwister()

POMDPs.actions(mdp::SimpleMDP) = DiscreteNonParametric([-1.0, 0.0, 1.0], [0.2, 0.3, 0.5])
POMDPs.isterminal(mdp::SimpleMDP, s) = s > 5
POMDPs.discount(mdp::SimpleMDP) = 1.0

function POMDPs.gen(mdp::SimpleMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    sp = s + a
    r = 10 - sp
    return (sp=sp, r=r)
end

mdp = SimpleMDP()

fixed_s = rand(initialstate(mdp))

import TISExperiments

N = 1000
c = 0.3
α = 0.1

tree_mdp = TISExperiments.construct_tree_amdp(rmdp, actions; reduction="sum")

run_baseline_and_treeIS(mdp, tree_mdp, fixed_s, actions, "/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/simple"; N, c, α)
