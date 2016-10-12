function [Lint_I, occ_I] = condition_joint_mark_intensity_on_discrete_state(xtrain, mark_spikes_to_linear_position_time_bins_index, is_state, sxker, mark_bins, linear_distance_bins, dt)
num_mark_bins = length(mark_bins);
num_state = length(is_state);
num_linear_distance_bins = length(linear_distance_bins);

mark_spikes_to_linear_position_time_bins_index_I = mark_spikes_to_linear_position_time_bins_index(ismember(mark_spikes_to_linear_position_time_bins_index, is_state));
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