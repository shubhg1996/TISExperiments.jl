using POMDPGym, POMDPs

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