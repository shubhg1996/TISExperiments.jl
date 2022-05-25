using AdversarialDriving
using AutomotiveSimulator
using AutomotiveVisualization
using POMDPStressTesting
using Parameters
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions
using Random

include("/home/users/shubhgup/Codes/Julia/AutonomousRiskFramework.jl/RiskSimulator.jl/src/scenarios/scenarios.jl")

@with_kw struct AutoRiskParams
    endtime::Real = 30 # Simulate end time
    ignore_sensors::Bool = true # Simulate sensor observations for agents
end;

@with_kw mutable struct AutoRiskSim <: GrayBox.Simulation
    t::Real = 0 # Current time
    params::AutoRiskParams = AutoRiskParams() # Parameters

    # Initial Noise
    init_noise_1 = Noise(pos=(0,0), vel=0)
    init_noise_2 = Noise(pos=(0,0), vel=0)

    # Driving scenario
    scenario::Scenario = scenario_t_head_on_turn(init_noise_1=init_noise_1, init_noise_2=init_noise_2)

    # Roadway from scenario
    # roadway::Roadway = multi_lane_roadway() # Default roadway
    roadway::Roadway = scenario.roadway # Default roadway

    # System under test, ego vehicle
    # sut = BlinkerVehicleAgent(get_urban_vehicle_1(id=1, s=5.0, v=15.0, noise=init_noise_1, roadway=roadway),
    # sut = BlinkerVehicleAgent(t_left_to_right(id=1, noise=init_noise_1, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors));
    sut = scenario.sut

    # Noisy adversary, vehicle
    # adversary = BlinkerVehicleAgent(get_urban_vehicle_2(id=2, s=25.0, v=0.0, noise=init_noise_2, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=0.0), noisy_observations=false));
    # adversary = BlinkerVehicleAgent(t_right_to_turn(id=2, noise=init_noise_2, roadway=roadway),
    # UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false));
    adversary = scenario.adversary

    # Adversarial Markov decision process
    problem::MDP = AdversarialDrivingMDP(sut, [adversary], roadway, 0.1)
    state::Scene = rand(initialstate(problem))
    prev_distance::Real = -1e10 # Used when agent goes out of frame

    # Noise distributions and disturbances (consistent with output variables in _logpdf)
    xposition_noise_veh::Distribution = Normal(0, 5) # Gaussian noise (notice larger σ)
    yposition_noise_veh::Distribution = Normal(0, 5) # Gaussian noise
    velocity_noise_veh::Distribution = Normal(0, 1) # Gaussian noise

    xposition_noise_sut::Distribution = Normal(0, 5) # Gaussian noise (notice larger σ)
    yposition_noise_sut::Distribution = Normal(0, 5) # Gaussian noise
    velocity_noise_sut::Distribution = Normal(0, 1) # Gaussian noise

    disturbances = scenario.disturbances # Initial 0-noise disturbance

    _logpdf::Function = (sample, state) -> 0 # Function for evaluating logpdf
end

##############################################################################
"""
Transition and Reward
"""

"""
Generate next state and reward for AST MDP (handles episodic reward problems). Overridden from `POMDPs.gen` interface.
"""
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG=Random.GLOBAL_RNG)
    @assert mdp.sim_hash == s.hash
    if mdp.t_index == 1 # initial state indication
        prev_distance = 0
    else
        prev_distance = BlackBox.distance(mdp.sim)
    end
    mdp.t_index += 1
    isa(a, ASTSeedAction) ? POMDPStressTesting.set_global_seed(a) : nothing
    hasproperty(mdp.sim, :actions) ? push!(mdp.sim.actions, a) : nothing

    # Step black-box simulation
    if mdp.params.episodic_rewards
        # Do not evaluate when problem has episodic rewards
        if a isa ASTSeedAction
            if mdp.params.pass_seed_action
                logprob = GrayBox.transition!(mdp.sim, a.seed)
            else
                logprob = GrayBox.transition!(mdp.sim)
            end
        elseif a isa ASTSampleAction
            logprob = GrayBox.transition!(mdp.sim, a.sample)
        end
        isevent::Bool = false
        miss_distance::Float64 = NaN
        rate::Float64 = NaN
    else
        if a isa ASTSeedAction
            if mdp.params.pass_seed_action
                (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim, a.seed)
            else
                (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim)
            end
        elseif a isa ASTSampleAction
            (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim, a.sample)
        end
        rate = BlackBox.rate(prev_distance, mdp.sim)
    end

    # Update state
    sp = ASTState(t_index=mdp.t_index, parent=s, action=a)
    mdp.sim_hash = sp.hash
    mdp.rate = rate
    sp.terminal = mdp.params.episodic_rewards ? false : BlackBox.isterminal(mdp.sim) # termination handled by end-of-rollout
    r::Float64 = reward(mdp, logprob, isevent, sp.terminal, miss_distance, rate)
    sp.q_value = r
    
    return (sp=sp, r=r)
end

function POMDPs.reward(mdp::ASTMDP, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real, rate::Real)
    r = 0.0
    if isterminal && isevent
        r += rate # R_E (additive)
    elseif isterminal && !isevent
        r += -miss_distance # Only add miss distance cost if is terminal and not an event.
    end
    
    if isterminal
        record(mdp, prob=exp(logprob), logprob=logprob, miss_distance=miss_distance, reward=r, event=isevent, terminal=isterminal, rate=rate)
    end

    if isterminal
        # end of episode
        record_returns(mdp)
    end

    return r
end


##############################################################################
"""
Graybox functions
"""

function GrayBox.environment(sim::AutoRiskSim)
   return GrayBox.Environment(
                            :vel_veh => sim.velocity_noise_veh,
                            :xpos_veh => sim.xposition_noise_veh,
                            :ypos_veh => sim.yposition_noise_veh,
                            :vel_sut => sim.velocity_noise_sut,
                            :xpos_sut => sim.xposition_noise_sut,
                            :ypos_sut => sim.yposition_noise_sut
                        )
end;

function GrayBox.transition!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    sim.t += 1 # sim.problem.dt # Keep track of time
    noise_veh = Noise(pos = (sample[:xpos_veh].value, sample[:ypos_veh].value), vel = sample[:vel_veh].value) # reversed to match local pedestrain frame
    noise_sut = Noise(pos = (sample[:xpos_sut].value, sample[:ypos_sut].value), vel = sample[:vel_sut].value)
    sim.disturbances[1] = BlinkerVehicleControl(noise=noise_sut)
    sim.disturbances[2] = typeof(sim.disturbances[2])(noise=noise_veh) # could be BlinkerVehicleControl or PedestrianControl

    # step agents: given MDP, current state, and current action (i.e. disturbances)
    (sim.state, r) = @gen(:sp, :r)(sim.problem, sim.state, sim.disturbances)

    # return log-likelihood of actions, summation handled by `logpdf()`
    return sim._logpdf(sample, sim.state)::Real
end

##############################################################################
"""
Blackbox functions
"""

function BlackBox.initialize!(sim::AutoRiskSim)
    sim.t = 0
    sim.problem = AdversarialDrivingMDP(sim.sut, [sim.adversary], sim.roadway, 0.1)
    sim.state = rand(initialstate(sim.problem))
    sim.disturbances = Disturbance[BlinkerVehicleControl(), typeof(sim.disturbances[2])()] # noise-less
    sim.prev_distance = -1e10
end

out_of_frame(sim) = length(sim.state.entities) < 2 # either agent went out of frame

function BlackBox.distance(sim::AutoRiskSim)
    if out_of_frame(sim)
        return sim.prev_distance
    else
        vehicle, sut = sim.state.entities
        pos1 = posg(vehicle)
        pos2 = posg(sut)
        return hypot(pos1.x - pos2.x, pos1.y - pos2.y)
    end
end

function BlackBox.isevent(sim::AutoRiskSim)
    if out_of_frame(sim)
        return false
    else
        vehicle, sut = sim.state.entities
        return collision_checker(vehicle, sut)
    end
end

function BlackBox.isterminal(sim::AutoRiskSim)
    return isterminal(sim.problem, sim.state) ||
           out_of_frame(sim) ||
           BlackBox.isevent(sim) ||
           sim.t ≥ sim.params.endtime
end

function BlackBox.evaluate!(sim::AutoRiskSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim)           # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim)            # Check event indication
    sim.prev_distance = d                            # Store previous distance
    return (logprob::Real, d::Real, event::Bool)
end


##############################################################################
"""
Scene stepping for self-noise in SUT
"""

# Step the scene forward by one timestep and return the next state
function AdversarialDriving.step_scene(mdp::AdversarialDriving.AdversarialDrivingMDP, s::Scene, actions::Vector{Disturbance}, rng::AbstractRNG = Random.GLOBAL_RNG)
    entities = []

    # Add noise in SUT
    update_adversary!(sut(mdp), actions[1], s)

    # Loop through the adversaries and apply the instantaneous aspects of their disturbance
    for (adversary, action) in zip(adversaries(mdp), actions[2:end])
        update_adversary!(adversary, action, s)
    end

    # Loop through the vehicles in the scene, apply action and add to next scene
    for (i, veh) in enumerate(s)
        m = model(mdp, veh.id)
        observe!(m, s, mdp.roadway, veh.id)
        a = rand(rng, m)
        bv = Entity(propagate(veh, a, mdp.roadway, mdp.dt), veh.def, veh.id)
        !end_of_road(bv, mdp.roadway, mdp.end_of_road) && push!(entities, bv)
    end
    isempty(entities) ? Scene(typeof(sut(mdp).get_initial_entity())) : Scene([entities...])
end

#################################################################################
"""
    AST.record(::ASTMDP, sym::Symbol, val)
Recard an ASTMetric specified by `sym`.
"""
function record(mdp::ASTMDP, sym::Symbol, val)
    if mdp.params.debug
        push!(getproperty(mdp.metrics, sym), val)
    end
end

function record(mdp::ASTMDP; prob=1, logprob=exp(prob), miss_distance=Inf, reward=-Inf, event=false, terminal=false, rate=-Inf)
    AST.record(mdp, :prob, prob)
    AST.record(mdp, :logprob, logprob)
    AST.record(mdp, :miss_distance, miss_distance)
    AST.record(mdp, :reward, reward)
    AST.record(mdp, :intermediate_reward, reward)
    AST.record(mdp, :rate, rate)
    AST.record(mdp, :event, event)
    AST.record(mdp, :terminal, terminal)
end

function record_returns(mdp::ASTMDP)
    # compute returns up to now.
    rewards = mdp.metrics.intermediate_reward
    G = returns(rewards, γ=discount(mdp))
    AST.record(mdp, :returns, G)
    mdp.metrics.intermediate_reward = [] # reset
end

function returns(R; γ=1)
    T = length(R)
    G = zeros(T)
    for t in reverse(1:T)
        G[t] = t==T ? R[t] : G[t] = R[t] + γ*G[t+1]
    end
    return G
end


function combine_ast_metrics(plannervec::Vector)
    return ASTMetrics(
        miss_distance=vcat(map(planner->planner.mdp.metrics.miss_distance, plannervec)...),
        rate=vcat(map(planner->planner.mdp.metrics.rate, plannervec)...),
        logprob=vcat(map(planner->planner.mdp.metrics.logprob, plannervec)...),
        prob=vcat(map(planner->planner.mdp.metrics.prob, plannervec)...),
        reward=vcat(map(planner->planner.mdp.metrics.reward, plannervec)...),
        intermediate_reward=vcat(map(planner->planner.mdp.metrics.intermediate_reward, plannervec)...),
        returns=vcat(map(planner->planner.mdp.metrics.returns, plannervec)...),
        event=vcat(map(planner->planner.mdp.metrics.event, plannervec)...),
        terminal=vcat(map(planner->planner.mdp.metrics.terminal, plannervec)...))
end

#################################################################################
"""
MDP generating function
"""

function gen_ast_mdp(scenario, noise_vars)
    sim = AutoRiskSim(
        scenario=scenario, 
        xposition_noise_veh=Normal(0, noise_vars[1]),
        yposition_noise_veh=Normal(0, noise_vars[2]),
        velocity_noise_veh=Normal(0, noise_vars[3]),
        xposition_noise_sut=Normal(0, noise_vars[4]),
        yposition_noise_sut=Normal(0, noise_vars[5]),
        velocity_noise_sut=Normal(0, noise_vars[6]),
    )
    
    mdp = ASTMDP{ASTSampleAction}(sim)
    return mdp, sim
end


