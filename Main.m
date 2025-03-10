clc;
clear;
close all;

Fun_name = 'F9';                    
SearchAgents = 30;                  
Max_EFs = 30000;                     

[lowerbound, upperbound, dimension, fitness] = fun_info(Fun_name);


if isscalar(lowerbound)
    lowerbound = lowerbound * ones(1, dimension);
end
if isscalar(upperbound)
    upperbound = upperbound * ones(1, dimension);
end


func_num = str2double(Fun_name(2:end)); 
fhd = @(x, func_num) fitness(x);       


[Best_score, Best_pos, curve] = MS_EMGO(SearchAgents, Max_EFs, lowerbound, upperbound, dimension, func_num, fhd);


disp(['The best solution obtained by MS_EMGO for ' Fun_name ' is : ', num2str(Best_pos)]);
disp(['The optimal value of the objective function found by MS_EMGO is : ', num2str(Best_score)]);

figure(func_num);
hold on;
threshold = 1e10; 
curve(curve > threshold | curve <= 0) = NaN;

       
curve = fillmissing(curve, 'previous', 'EndValues', 'nearest') + 1;
curve = fillmissing(curve, 'next', 'EndValues', 'nearest') + 1;

x_fes = linspace(1, Max_EFs, Max_EFs);
Convergence_curve_MS_EMGO_interp = interp1(1:length(curve), curve, x_fes, 'linear', 'extrap');
plot(x_fes, Convergence_curve_MS_EMGO_interp, 'g', 'LineWidth', 1.5);

set(gca, 'Layer', 'top'); 

        xlabel('评估次数');
        ylabel('目标函数值');
        title(sprintf('Function F%d: Convergence of  MS_EMGO ', func_num));
        
        legend({'MS_EMGO (Green)'}, ...
                'Location', 'best',...
                'NumColumns', 5);