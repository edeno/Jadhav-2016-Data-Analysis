function [Lint_I, occ_I] = condition_Lambda_on_discrete_state(xtrain, procInd1, is_state, sxker, mark_bins, linear_distance_bins, dt)
num_mark_bins = length(mark_bins);
num_state = length(is_state);
num_linear_distance_bins = length(linear_distance_bins);

procInd1_I = procInd1(ismember(procInd1, is_state));
occ_I = normpdf(linear_distance_bins' * ones(1, num_state), ...
    ones(num_linear_distance_bins, 1) * xtrain(is_state), sxker) * ...
    ones(num_state, num_mark_bins);
Xnum_I = normpdf(linear_distance_bins' * ones(1, length(xtrain(procInd1_I))), ...
    ones(num_linear_distance_bins, 1) * xtrain(procInd1_I), ...
    sxker);

%Xnum: Gaussian kernel estimators for position
Lint_I = sum(Xnum_I, 2) ./ occ_I(:, 1) ./ dt; %integral
Lint_I = Lint_I ./ sum(Lint_I);
end