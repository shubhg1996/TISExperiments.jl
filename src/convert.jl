# Builds a Tree MDP from POMDPGym.RMDP
@export function construct_tree_rmdp(rmdp, distribution; reduction="max")
    return TreeMDP(rmdp, 1.0, [], [], (m, s) -> distribution, reduction)
end

# Builds a Tree MDP from POMDPGym.AMDP
@export function construct_tree_amdp(amdp, distribution; reduction="sum")
    return TreeMDP(amdp, 1.0, [], [], distribution, reduction)
end