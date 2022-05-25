@export function plot_metrics(costs, name, idx; weights=nothing, alpha=1e-3, plots = [])
    if length(size(costs)) == 2
        N_seeds, N_samples = size(costs)
    else
        N_samples = length(costs)
        N_seeds = 1
        costs = reshape(costs, (1, N_samples))
        weights = reshape(weights, (1, N_samples))
    end
    
    costs = Float64.(costs)
    weights = weights===nothing ? ones(size(costs)) : Float64.(weights)
    
    
    if length(plots) == 0
        plots = [
            plot(title = "Mean", legend=:bottomright), 
            plot(title = "VaR", legend=:bottomright), 
            plot(title = "CVaR", legend=:bottomright), 
            plot(title = "Worst", legend=:bottomright)
        ]
        
        lim_costs = vcat(costs...)
        lim_weights = vcat(weights...)
        
        limit_metrics = IWRiskMetrics(lim_costs, lim_weights, alpha);
        label = "Limit $(length(lim_costs))"
        plot!(plots[1], [1, N_samples], ones(2)*limit_metrics.mean, color=0, label=label, linestyle=:dash)
        plot!(plots[2], [1, N_samples], ones(2)*limit_metrics.var, color=0, label=label, linestyle=:dash)
        plot!(plots[3], [1, N_samples], ones(2)*limit_metrics.cvar, color=0, label=label, linestyle=:dash)
        plot!(plots[4], [1, N_samples], ones(2)*limit_metrics.worst, color=0, label=label, linestyle=:dash)
    end
    
    N_range = [N for N=1:10:N_samples]
    seed_metrics = zeros(4, N_seeds, length(N_range))
    
    for i_sample=1:length(N_range)
        N = N_range[i_sample]
        for i_seed=1:N_seeds
            metrics = IWRiskMetrics(costs[i_seed, 1:N], weights[i_seed, 1:N], alpha);
            seed_metrics[1, i_seed, i_sample] = metrics.mean
            seed_metrics[2, i_seed, i_sample] = metrics.var
            seed_metrics[3, i_seed, i_sample] = metrics.cvar
            seed_metrics[4, i_seed, i_sample] = metrics.worst
        end
    end
    u_metrics = maximum(seed_metrics, dims=2)
    l_metrics = minimum(seed_metrics, dims=2)
    plot!(plots[1], N_range, l_metrics[1, 1, :], fillrange=u_metrics[1, 1, :], fillalpha = 0.35, color=idx, label=name)
    plot!(plots[2], N_range, l_metrics[2, 1, :], fillrange=u_metrics[2, 1, :], fillalpha = 0.35, color=idx, label=name)
    plot!(plots[3], N_range, l_metrics[3, 1, :], fillrange=u_metrics[3, 1, :], fillalpha = 0.35, color=idx, label=name)
    plot!(plots[4], N_range, l_metrics[4, 1, :], fillrange=u_metrics[4, 1, :], fillalpha = 0.35, color=idx, label=name)
    
    return plots
end