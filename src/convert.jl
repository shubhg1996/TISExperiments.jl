function construct_tree_rmdp(rmdp, distribution; reduction="max")
    return TreeMDP(rmdp, 1.0, [], [], (m, s) -> distribution, reduction)
end

function construct_tree_amdp(amdp, distribution; reduction="sum")
    return TreeMDP(amdp, 1.0, [], [], distribution, reduction)
end

function run_baseline_and_treeIS(mdp, tree_mdp, fixed_s, disturbance; N=1000, c=0.3, α=0.1, thresh=0.5, β=1.0, γ=0.0, baseline=true)
    if baseline
        # BASELINE
        @show "Executing baseline"

        baseline_costs = [sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(disturbance(mdp, s))), fixed_s)[:r])) for _ in 1:N*10]
    else
        baseline_costs = []
    end

    # MCTS
    @show "Executing Tree-IS"

    planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)

    a, info = action_info(planner, TreeImportanceSampling.TreeState(fixed_s); tree_in_info=true, β, γ)

    return (baseline_costs, []), (planner.mdp.costs, [], planner.mdp.IS_weights, info[:tree]), planner
end

function evaluate_metrics(costs; weights=nothing, alpha_list=[1e-4])
    costs = [Float64(cost) for cost in costs]
    if weights===nothing
        weights = ones(length(costs))
    else
        weights = [Float64(weight) for weight in weights]
    end
    for alpha in alpha_list
        print("\n\nAlpha: ", alpha)
        metrics = IWRiskMetrics(costs, weights, alpha);
        print("\nMean: $(metrics.mean), VaR: $(metrics.var), CVaR: $(metrics.cvar), Worst: $(metrics.worst)")
    end
end
