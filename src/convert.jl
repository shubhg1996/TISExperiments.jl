function construct_tree_rmdp(rmdp, distribution; reduction="max")
    return TreeMDP(rmdp, 1.0, [], [], (m, s) -> distribution, reduction)
end

function construct_tree_amdp(amdp, distribution; reduction="sum")
    return TreeMDP(amdp, 1.0, [], [], distribution, reduction)
end
