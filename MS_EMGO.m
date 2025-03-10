function [best_cost, best_M, Convergence_curve] = MS_EMGO(SearchAgents_no, MaxFEs, lb, ub, dim, func_num, fhd)
    FEs = 0;
    best_cost = inf;
    best_M = zeros(1, dim);
    M = initialization(SearchAgents_no, dim, ub, lb);
    costs = zeros(1, SearchAgents_no);

    for i = 1:SearchAgents_no
        costs(i) = feval(fhd, M(i, :)', func_num);
        FEs = FEs + 1;
        if costs(i) < best_cost
            best_M = M(i, :);
            best_cost = costs(i);
        end
    end

    Convergence_curve = zeros(1, MaxFEs);
    
    w_max = 2.5;
    w_min = 0.1; 
    
   
    while FEs < MaxFEs
        
        w = w_max - ((w_max - w_min) * (FEs / MaxFEs)^2);
        
       
        D_wind = mean(M - best_M, 1);
        
        for i = 1:SearchAgents_no
            
            idxs = randperm(SearchAgents_no, 3);
            while any(idxs == i)
                idxs = randperm(SearchAgents_no, 3);
            end
            F = 0.7 + 0.3 * (1 - (costs(i) - best_cost) / max(eps, (max(costs) - min(costs))));
            CR = 0.9 - 0.4 * (1 - F);

            step_size = 0.5 + 0.5 * (1 - (costs(i) - best_cost) / (max(costs) - min(costs) + eps));
       
            if rand() < 0.5
                mutant = best_M + F * (M(idxs(1), :) - M(idxs(2), :));
            else
                mutant = M(i, :) + F * (best_M - M(i, :)) + F * (M(idxs(1), :) - M(idxs(2), :)); 
            end
            mutant = mirror_boundary(mutant, ub, lb);
            
            crossover = rand(1, dim) < CR;
            trial = M(i, :) + step_size * (rand(1, dim) - 0.5) .* D_wind;
            trial(crossover) = mutant(crossover);
            if rand() < 0.5
                trial = trial + w * levy(dim) .* (trial - best_M); 
            end
            trial = mirror_boundary(trial, ub, lb);
            
            trial_cost = feval(fhd, trial', func_num);
            FEs = FEs + 1;
            if trial_cost < costs(i)
                M(i, :) = trial;
                costs(i) = trial_cost;
            end
            if trial_cost < best_cost
                best_M = trial;
                best_cost = trial_cost;
            end
            if FEs >= MaxFEs, break; end
        end
        
        diversity = mean(sqrt(sum((M - mean(M)).^2, 2)));
        dynamic_threshold = 1e-4 * (1 - FEs / MaxFEs);
        if diversity < dynamic_threshold
            for j = 1:ceil(0.1 * SearchAgents_no)
                M(j, :) = best_M + w * levy(dim) .* (ub - lb); 
                M(j, :) = mirror_boundary(M(j, :), ub, lb);
                costs(j) = feval(fhd, M(j, :)', func_num);
                FEs = FEs + 1;
                if FEs >= MaxFEs, break; end
            end
        end
        
        % 小范围小测操作
        if rand() < 0.1
            best_M = local_search(best_M, ub, lb, fhd, func_num);
        end
        
        % 更新收敛曲线
        Convergence_curve(FEs) = best_cost;
    end
end

function [M] = initialization(SearchAgents_no, dim, ub, lb)
    M = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

function newX = mirror_boundary(X, ub, lb)
    if isscalar(ub)
        ub = ub * ones(size(X));
    end
    if isscalar(lb)
        lb = lb * ones(size(X));
    end

    X(X > ub) = ub(X > ub) - mod(X(X > ub) - ub(X > ub), (ub(X > ub) - lb(X > ub)));
    X(X < lb) = lb(X < lb) + mod(lb(X < lb) - X(X < lb), (ub(X < lb) - lb(X < lb)));
    newX = X;
end

function L = levy(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    L = u ./ abs(v).^(1 / beta);
end

function local_best = local_search(current_best, ub, lb, fhd, func_num)
    step_size = 0.01 * (ub - lb);
    perturb = (randn(size(current_best)) .* step_size);
    trial = current_best + perturb;
    trial = mirror_boundary(trial, ub, lb);
    trial_cost = feval(fhd, trial', func_num);
    if trial_cost < feval(fhd, current_best', func_num)
        local_best = trial;
    else
        local_best = current_best;
    end
end