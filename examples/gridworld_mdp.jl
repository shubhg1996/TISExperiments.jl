using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies
using Crux, Flux, BSON, StaticArrays, Random
include("generic_dnp.jl")

# Basic MDP
Random.seed!(42)
tprob = 0.99
randcosts = Dict(POMDPGym.GWPos(i,j) => rand() for i = 1:10, j=1:10)
# zerocosts = Dict(POMDPGym.GWPos(i,j) => 0.0 for i = 1:10, j=1:10)
mdp = GridWorldMDP(costs=randcosts, cost_penalty=0.1, tprob=1.0)
all_actions = actions(mdp)

# # Learn a policy that solves it
# policy = DiscreteNetwork(Chain(x -> (x .- 5f0) ./ 5f0, Dense(2, 32, relu), Dense(32, 4)), [:up, :down, :left, :right]);
# policy = solve(DQN(π=policy, S=state_space(mdp), N=100000, ΔN=4, buffer_size=10000, log=(;period=5000)), mdp);
# atable = Dict(s => action(policy, [s...]) for s in states(mdp.g));
# BSON.@save "policies/gridworld_policy_table_tprob_1.bson" atable

atable = BSON.load("policies/gridworld_policy_table_tprob_1.bson")[:atable]

function gen_gridworld_mdp(; penalty_mul=10, maxT=20)
    # Define the adversarial mdp
    adv_rewards = deepcopy(randcosts)
    # adv_rewards = deepcopy(zerocosts)
    for (k,v) in adv_rewards
        if k in keys(mdp.g.rewards)
            if v < 0
                adv_rewards[k] = 0
            else
                adv_rewards[k] += penalty_mul*v
            end
        else
            adv_rewards[k] += penalty_mul*v
        end
    end
    
    amdp = GridWorldMDP(rewards=adv_rewards, tprob=1., discount=1., terminate_from=mdp.g.terminate_from)
    
    costfn(m, s, sp) = POMDPs.reward(m.amdp, s)
    policy = NoisyGridWorldPolicy()
    return adv_rewards, RMDP(amdp, policy, costfn, false, 1, maxT, :action)
end

# Define action probability for the adv_mdp
action_probability(s, a) = (a == atable[s][1]) ? tprob : ((1. - tprob) / (length(all_actions) - 1.))

struct NoisyGridWorldPolicy <: POMDPs.Policy end;
POMDPs.action(policy::NoisyGridWorldPolicy, s) = sample(all_actions, [action_probability(s, a) for a in all_actions])

function disturbance(m::GridWorldMDP, s)
    xs = POMDPs.actions(m, s)
    ps = [action_probability(m, s, x) for x in xs]
    ps ./= sum(ps)
    px = GenericDiscreteNonParametric(xs, ps)
    return px
end

function POMDPs.actions(mdp::RMDP) end
POMDPs.actions(mdp::RMDP, s) = disturbance(mdp.amdp, POMDPGym.get_s(mdp, s))

# Redefinition with direct action disturbances
function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    if mdp.disturbance_type == :arg
        sp, r = gen(mdp.amdp, POMDPGym.get_s(mdp, s), action(mdp.π, POMDPGym.get_s(mdp, s)), x, rng; kwargs...)
    elseif mdp.disturbance_type == :noise
        sp, r = gen(mdp.amdp, POMDPGym.get_s(mdp, s), action(mdp.π, POMDPGym.get_s(mdp, s) .+ x), rng; kwargs...)
    elseif mdp.disturbance_type == :both
        sp, r = gen(mdp.amdp, POMDPGym.get_s(mdp, s), action(mdp.π, POMDPGym.get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
    elseif mdp.disturbance_type == :action
        sp, r = gen(mdp.amdp, POMDPGym.get_s(mdp, s), x, rng; kwargs...)
    else
        @error "Unrecognized disturbance type $(mdp.disturbance_type)"
    end
        
    if mdp.include_time_in_state
        t = s[1]
        sp = [t+mdp.dt, sp...]
    end
    (sp=sp, r=mdp.cost_fn(mdp, s, sp))
end