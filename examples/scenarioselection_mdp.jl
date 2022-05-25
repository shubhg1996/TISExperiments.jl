using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, POMDPModelTools
using Parameters
using Random

include("ast_mdp.jl")
include("generic_dnp.jl")

DecisionState_limits = [1, 3, 2, 2]

@with_kw struct DecisionState 
    scenario_type::Any = nothing # scenario type
    noise_params::Vector{Float64} = [] # noise parameters
    init_sut::Vector{Float64} = [] # Initial conditions SUT
    init_adv::Vector{Float64} = [] # Initial conditions Adversary
    done::Bool = false
end

# initial state constructor
function DecisionState(s::DecisionState, val)
    scenario_type = s.scenario_type
    noise_params = s.noise_params
    init_sut = s.init_sut
    init_adv = s.init_adv
    done = false
    if s.scenario_type === nothing
        scenario_type = val
    elseif length(s.noise_params) < DecisionState_limits[2]
        noise_params = [s.noise_params..., val]
    elseif length(s.init_sut) < DecisionState_limits[3]
        init_sut = [s.init_sut..., val]
    elseif length(s.init_adv) < DecisionState_limits[4]
        init_adv = [s.init_adv..., val]
    else
        done = true
    end
    
    return DecisionState(scenario_type, noise_params, init_sut, init_adv, done)
end

function check_full(s::DecisionState)
    return !(s.scenario_type === nothing) && (length(s.noise_params) == DecisionState_limits[2]) && (length(s.init_sut) == DecisionState_limits[3]) && (length(s.init_adv) == DecisionState_limits[4])
end
   
struct UniformScenarioSelectionPolicy <: POMDPs.Policy end;
POMDPs.action(policy::UniformScenarioSelectionPolicy, s::DecisionState) = rand(get_actions(s))

get_sut_pos(s::DecisionState) = Float64(s.init_sut[1])
get_adv_pos(s::DecisionState) = Float64(s.init_adv[1])
get_sut_vel(s::DecisionState) = Float64(s.init_sut[2])
get_adv_vel(s::DecisionState) = Float64(s.init_adv[2])

# Define the system to test
system = IntelligentDriverModel()    

# Evaluates a scenario using AST
# Returns: scalar risk if failures were discovered, 0 if not, -10.0 if an error occured during search

function create_noise_cpd(n_levels, level_width)
    level_options = rand(n_levels, level_width)
    values = ones(level_width^n_levels, 6);
    for i=1:level_width^n_levels
        elems = rand(1:6, 3)
        values[i, elems[1]] = 5.0
        values[i, elems[2]] = 5.0
        values[i, elems[3]] = 5.0
    end
    values = reshape(values, ([level_width for i=1:n_levels]..., 6))

    function get_noise(noise_params)
        N_params = length(noise_params)
        noise_params = reshape(noise_params, (N_params, 1))
        diff = abs.(level_options .- noise_params)
        idx = argmin(diff, dims=2)
        value = values[[idx[i][2] for i=1:N_params]..., :]
        return value
    end

    get_options(k) = DiscreteNonParametric(level_options[k, :], ones(level_width)/level_width)
    return get_noise, get_options, values
end

get_noise, get_options, noise_values = create_noise_cpd(3, 5)

get_scenario_types() = GenericDiscreteNonParametric([CROSSING, T_HEAD_ON, T_LEFT, STOPPING, MERGING, CROSSWALK], ones(6)/6)

function get_scenario_feat_options(scenario_enum, feat; num=1)
    scenario_ranges = get_scenario_options(scenario_enum)
    opt_range = scenario_ranges[feat]
    dx = (opt_range[2] - opt_range[1])/num
    opts = [opt_range[1] + i*dx for i=1:num]
    return DiscreteNonParametric(opts, ones(num)/num)
end

function get_actions(s::DecisionState)
    if s.scenario_type === nothing
        return get_scenario_types()
    elseif length(s.noise_params) < DecisionState_limits[2]
        return get_options(length(s.noise_params)+1)
    elseif length(s.init_sut) < DecisionState_limits[3]-1
        return get_scenario_feat_options(s.scenario_type, "s_sut")
    elseif length(s.init_sut) < DecisionState_limits[3]
        return get_scenario_feat_options(s.scenario_type, "v_sut")
    elseif length(s.init_adv) < DecisionState_limits[4]-1
        return get_scenario_feat_options(s.scenario_type, "s_adv")
    elseif length(s.init_adv) < DecisionState_limits[4]
        return get_scenario_feat_options(s.scenario_type, "v_adv")
    else
        return get_scenario_types()
    end
end

function eval_AST(s::DecisionState; return_sim=false)
    @assert check_full(s)
    scenario = get_scenario(s.scenario_type; s_sut=get_sut_pos(s), s_adv=get_adv_pos(s), v_sut=get_sut_vel(s), v_adv=get_adv_vel(s))
    noise_vars = get_noise(s.noise_params)
    mdp, sim = gen_ast_mdp(scenario, noise_vars)
    
    if return_sim
        return mdp, sim
    else
        r = POMDPs.simulate(RolloutSimulator(), mdp, RandomPolicy(mdp))
        return max(r, 0)
    end
end

# The scenario decision mdp type
mutable struct ScenarioSearch <: MDP{DecisionState, Any}
    discount_factor::Float64 # disocunt factor
end

function POMDPs.reward(mdp::ScenarioSearch, state::DecisionState)
    if !check_full(state)
        r = 0
    else
        r = eval_AST(state)
    end
    return r
end

POMDPs.reward(mdp::ScenarioSearch, state::DecisionState, action::Any) = POMDPs.reward(mdp, state)

function POMDPs.initialstate(mdp::ScenarioSearch) # rng unused.
    return POMDPModelTools.ImplicitDistribution((rng) -> DecisionState())
end

# # Base.convert(::Type{Int64}, x) = x
# # convert(::Type{Union{Float64, Nothing}}, x) = x

function POMDPs.gen(m::ScenarioSearch, s::DecisionState, a, rng)
    sp = DecisionState(s, a)
    r = POMDPs.reward(m, s, a)
    return (sp=sp, r=r)
end

function POMDPs.isterminal(mdp::ScenarioSearch, s::DecisionState)
    return s.done
end

POMDPs.discount(mdp::ScenarioSearch) = mdp.discount_factor

function POMDPs.actions(mdp::ScenarioSearch, s::DecisionState)
    return get_actions(s)
end

function disturbance(m::ScenarioSearch, s)
    return POMDPs.actions(m, s)
end

######################################################################################
"""
Create RMDP
"""

function gen_scenarioselection_mdp()
    maxT = 20
    mdp = ScenarioSearch(1.0)
    policy = UniformScenarioSelectionPolicy()
    costfn(m, s, sp) = POMDPs.reward(m.amdp, s)
    return RMDP(mdp, policy, costfn, false, 1, maxT, :action)
end

# function rollout(mdp::ScenarioSearch, s::DecisionState, d::Int64)
#     if d == 0 || isterminal(mdp, s)
#         return 0.0
#     else
#         a = rand(POMDPs.actions(mdp, s))

#         (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
#         q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

#         return q_value
#     end
# end