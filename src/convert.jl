function construct_tree_rmdp(rmdp, distribution; reduction="max")
    return TreeMDP(rmdp, 1.0, [], [], (m, s) -> distribution, reduction)
end

function construct_tree_amdp(amdp, distribution; reduction="sum")
    return TreeMDP(amdp, 1.0, [], [], distribution, reduction)
end

function run_baseline_and_treeIS(mdp, tree_mdp, fixed_s, disturbance; N=1000, c=0.3, α=0.1)
    # BASELINE
    @show "Executing baseline"

    baseline_costs = [sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(disturbance(mdp, s))), fixed_s)[:r])) for _ in 1:N*100]

    # MCTS
    @show "Executing Tree-IS"

    planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)

    a, info = action_info(planner, TreeImportanceSampling.TreeState(fixed_s); tree_in_info=true)

    return (baseline_costs, []), (planner.mdp.costs, [], planner.mdp.IS_weights, info[:tree]), planner
end
