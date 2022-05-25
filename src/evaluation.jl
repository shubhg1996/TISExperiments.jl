@export function gen_baseline_costs(mdp, fixed_s; N=1000, N_seeds=10)
    all_costs = zeros(N_seeds, N)
    for i_seed=1:N_seeds
        Random.seed!(i_seed)
        @showprogress for i=1:N
            cost = sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(actions(mdp, s))), fixed_s)[:r]))
            all_costs[i_seed, i] = cost
        end
    end
    return all_costs
end

@export function gen_treeis_costs(mdp, fixed_s, params; N=1000, N_seeds=10)
    all_costs = zeros(N_seeds, N)
    all_weights = zeros(N_seeds, N)
    for i_seed=1:N_seeds
        Random.seed!(i_seed)
        costs, weights, stored_tree = update_treeis(mdp, fixed_s, params; N=N)
        all_costs[i_seed, :] = costs
        all_weights[i_seed, :] = exp.(weights)
    end
    return all_costs, all_weights
    
    
end

@export function update_treeis(mdp, fixed_s, params; N=1000, stored_tree=nothing)
    if stored_tree === nothing
        tree_mdp = construct_tree_amdp(mdp, (m, s) -> actions(m, s); reduction="sum")
        planner = TreeImportanceSampling.mcts_isdpw(tree_mdp, params; N=N)
        tree_fixed_s = TreeImportanceSampling.TreeState(fixed_s);
    else
        tree_fixed_s, planner = stored_tree
    end
    
    a, info = TreeImportanceSampling.action_info(planner, tree_fixed_s; tree_in_info=true, params=params)

    return planner.mdp.costs, planner.mdp.IS_weights, (tree_fixed_s, planner)
end


# General handler for algorithm execution
@export function run_baseline_and_treeIS(mdp, fixed_s, disturbance; baseline=false, N=1000, c=0.0, α=0.1, kwargs...)
    baseline_output = ([], [])
    tis_output = ([], [], [])
    if baseline
        # BASELINE
        @show "Executing baseline"

        baseline_costs = [sum(collect(simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> rand(disturbance(mdp, s))), fixed_s)[:r])) for _ in 1:N]
        baseline_output = (baseline_costs, [])
        planner = nothing
    else
        # MCTS
        @show "Executing Tree-IS"

        tree_mdp = construct_tree_amdp(mdp, disturbance; reduction="sum")

        planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)

        a, info = action_info(planner, TreeImportanceSampling.TreeState(fixed_s); tree_in_info=true, kwargs...)
        tis_output = (planner.mdp.costs, [], planner.mdp.IS_weights, info[:tree])
    end

    return baseline_output, tis_output, planner
end

# Compute metrics from the cost distribution
@export function evaluate_metrics(costs; weights=nothing, alpha_list=[1e-4])
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

# Compute the error and save the results out to a csv.
@export function compute_error(α, mc_samps, mc_weights, alg_samps, alg_weights)
    ground_truth = IWRiskMetrics(mc_samps, mc_weights, α)

    # Take subsets of the samples and compute the min error.
    min_var_rel, min_cvar_rel = Inf, Inf
    l = length(alg_samps)
    p = l ÷ 10
    for n in p:p:l
        alg_risk_metrics = IWRiskMetrics(alg_samps[1:n], alg_weights[1:n], α)
        var_rel_err = abs(ground_truth.var - alg_risk_metrics.var) / ground_truth.var
        cvar_rel_err = abs(ground_truth.cvar - alg_risk_metrics.cvar) / ground_truth.cvar
        min_var_rel = min(min_var_rel, var_rel_err)
        min_cvar_rel = min(min_cvar_rel, cvar_rel_err)
    end
    return min_var_rel, min_cvar_rel
end

# Runs experiments with Tree IS parameters
@export function run_grid_search(mdp, fixed_s, disturbance, mc_samps, mc_weights, path; N_l=[10_000, 50_000], α_l=[1e-2, 1e-3, 1e-4], β_a_l=[0.1, 0.5, 0.9], β_b_l=[0.1, 0.5, 0.9], schedule_l = [0.1, 0.5, Inf], floor_l = [0.01, 0.1, 0.5, 0.9], save_every=5)
    # Iterate over all params and do run_baseline_and_treeIS()
    df = DataFrame(alpha=Float64[], N=Integer[], beta=Float64[], gamma=Float64[], schedule=Float64[], floor=Float64[],
                    var_err_1em2=Float64[], cvar_err_1em2=Float64[],
                    var_err_1em3=Float64[], cvar_err_1em3=Float64[],
                    var_err_1em4=Float64[], cvar_err_1em4=Float64[])
    all_params = [p for p in Iterators.product(α_l, N_l, β_a_l, β_b_l, schedule_l, floor_l)]
    for (idx, params) in enumerate(all_params)
        println("Running ", idx, " / ", length(all_params))
        α, N, β_a, β_b, schedule, floor = params
        β = β_a*(1.0 - β_b)
        γ = β_a*β_b
        try
            results_baseline, results_alg, planner = run_baseline_and_treeIS(mdp, fixed_s, disturbance; N, c=0.0, α, β, γ, schedule, uniform_floor=floor, baseline=false)

            alg_samps = [Float64(samp) for samp in results_alg[1]]
            alg_weights = [exp(logwt) for logwt in results_alg[3]]

            var_rel_1em2, cvar_rel_1em2 = compute_error(1e-2, mc_samps, mc_weights, alg_samps, alg_weights)
            var_rel_1em3, cvar_rel_1em3 = compute_error(1e-3, mc_samps, mc_weights, alg_samps, alg_weights)
            var_rel_1em4, cvar_rel_1em4 = compute_error(1e-4, mc_samps, mc_weights, alg_samps, alg_weights)
            push!(df,
                  [α,
                   N,
                   β,
                   γ,
                   schedule,
                   floor,
                   var_rel_1em2, cvar_rel_1em2,
                   var_rel_1em3, cvar_rel_1em3,
                   var_rel_1em4, cvar_rel_1em4]);
            if idx % save_every == 0 || idx==length(all_params)
                CSV.write(string("$(path)_tis_", idx, ".csv"), df)
            end
        catch e
            println("Exception thrown for ", params, ": ", e)
            continue
        end
    end
end


@export function run_ablation(mdp, fixed_s, disturbance, mc_samps, mc_weights, path; N=10_000, α=1e-4, β=0.25, γ=0.45, schedule_l = [0.1, Inf], floor_l = [0.01, 1e-7], runs=10,save_every=5)
    # Iterate over all params and do run_baseline_and_treeIS()
    df = DataFrame(schedule=Float64[], floor=Float64[], run=Int[],
                    var_err_1em2=Float64[], cvar_err_1em2=Float64[],
                    var_err_1em3=Float64[], cvar_err_1em3=Float64[],
                    var_err_1em4=Float64[], cvar_err_1em4=Float64[])
    run_l = [i for i=1:runs]
    all_params = [p for p in Iterators.product(schedule_l, floor_l, run_l)]
    for (idx, params) in enumerate(all_params)
        println("Running ", idx, " / ", length(all_params))
        schedule, floor, run = params
        try
            results_baseline, results_alg, planner = run_baseline_and_treeIS(mdp, fixed_s, disturbance; N, c=0.0, α, β, γ, schedule, uniform_floor=floor, baseline=false)

            alg_samps = [Float64(samp) for samp in results_alg[1]]
            alg_weights = [exp(logwt) for logwt in results_alg[3]]

            var_rel_1em2, cvar_rel_1em2 = compute_error(1e-2, mc_samps, mc_weights, alg_samps, alg_weights)
            var_rel_1em3, cvar_rel_1em3 = compute_error(1e-3, mc_samps, mc_weights, alg_samps, alg_weights)
            var_rel_1em4, cvar_rel_1em4 = compute_error(1e-4, mc_samps, mc_weights, alg_samps, alg_weights)
            push!(df,
                  [schedule,
                   floor,
                   run,
                   var_rel_1em2, cvar_rel_1em2,
                   var_rel_1em3, cvar_rel_1em3,
                   var_rel_1em4, cvar_rel_1em4]);
            print('\n', schedule, cvar_rel_1em2, cvar_rel_1em3, cvar_rel_1em4)
            if idx % save_every == 0 || idx==length(all_params)
                CSV.write(string("$(path)_tis_", idx, ".csv"), df)
            end
        catch e
            println("Exception thrown for ", params, ": ", e)
            continue
        end
    end
end
