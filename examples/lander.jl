using Pkg
Pkg.activate("/home/users/shubhgup/Codes/Julia/TISExperiments.jl")
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using BSON
using SARSOP, Random
using BOMCP
using BeliefUpdaters
using FileIO
include("lander_pomdp.jl")
Random.seed!(42)
pomdp = LunarLander()

belief_updater = EKFUpdater(pomdp, pomdp.Q.^2, pomdp.R.^2)
rollout_policy = LanderPolicy(pomdp)

function BOMCP.vectorize!(v, dims, x::MvNormal)
    v = copy(mean(x))
    return v
end

action_selector = BOActionSelector(3, # action dims
                                6, #belief dims
                                false, #discrete actions
                                kernel_params=[log(5.0), 0.0],
                                k_neighbors = 5,
                                belief_λ = 0.5,
                                lower_bounds = [-10.0, 0.0, -0.5],
                                upper_bounds = [10.0, 15.0, 0.5],
                                buffer_size=100,
                                initial_action=rollout_policy
                                )


solver = BOMCPSolver(action_selector, belief_updater,
                    depth=250, n_iterations=100,
                    exploration_constant=50.0,
                    k_belief = 2.0,
                    alpha_belief = 0.1,
                    k_action = 3.,
                    alpha_action = 0.25,
                    estimate_value=BOMCP.RolloutEstimator(rollout_policy),
                    )

policy = POMDPs.solve(solver, pomdp)

# policy = load_policy(pomdp, "policy.out")

# dt = 0.1 # Do not change
# maxT = 2.0 # Do not change

# function costfn(x)
#     st, a, sp, o, r = x
#     randcosts[st[2].pos] - r + randcosts_a[a]
# end
# amdp = RPOMDP(pomdp=pomdp, π=policy, updater=DiscreteUpdater(pomdp), initial_belief_distribution = uniform_belief(pomdp), cost_fn=costfn, dt=dt, maxT=maxT)

# function action_probability(mdp, s, x)
#     d = POMDPs.observation(mdp.pomdp, action(mdp.π, s[3]), s[2])
#     d.probs[findfirst(d.vals .== x)]
# end

# function disturbance(m::typeof(amdp), s)
#     xs = POMDPs.actions(m, s)
#     ps = [action_probability(m, s, x) for x in xs]
#     ps ./= sum(ps)
#     px = DiscreteNonParametric(xs, ps)
#     return px
# end

# fixed_s = rand(initialstate(amdp))

# import TISExperiments

# N = 10_000
# c = 0.0
# α = 0.01

# # β = 0.25
# # γ = 0.25
# β = 0.00
# γ = 0.00

# schedule = 0.1

# baseline = true

# path = "data/rocksample"

# tree_mdp = TISExperiments.construct_tree_amdp(amdp, disturbance; reduction="sum")

# results_baseline, results_tis, planner = TISExperiments.run_baseline_and_treeIS(amdp, tree_mdp, fixed_s, disturbance; N=N, c=c, α=α, β=β, γ=γ, schedule=schedule, baseline=baseline);

# if baseline
#     print("Baseline metrics")

#     TISExperiments.evaluate_metrics(results_baseline[1]; alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
# else
#     print("\nTIS metrics: N=$(N), c=$(c), α=$(α), β=$(β)), γ=$(γ)")

#     TISExperiments.evaluate_metrics(results_tis[1]; weights=exp.(results_tis[3]), alpha_list=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
# end

# if baseline
#     save("$(path)_baseline_$(N).jld2", Dict("risks:" => results_baseline[1], "states:" => results_baseline[2]))
# else
#     save("$(path)_mcts_IS_$(N).jld2", Dict("risks:" => results_tis[1], "states:" => results_tis[2], "IS_weights:" => results_tis[3], "tree:" => results_tis[4]))
# end
