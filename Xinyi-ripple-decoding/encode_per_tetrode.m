function [mark_spikes_to_linear_position_time_bins_index_I, gaussian_kernel_position_estimator_I, Lint_I] = encode_per_tetrode( ...
    mark_spikes_to_linear_position_time_bins_index, is_state,  occ_I, dt, linear_distance_bins, xtrain, sxker)

Xnum_I = normpdf(linear_distance_bins' * ones(1, length(procInd1_Ia)), ...
    ones(length(linear_distance_bins), 1) * xtrain(procInd1_Ia), ...
position_time_bin_in_state = ismember(mark_spikes_to_linear_position_time_bins_index, is_state);
mark_spikes_to_linear_position_time_bins_index_Ia = mark_spikes_to_linear_position_time_bins_index(position_time_bin_in_state);
mark_spikes_to_linear_position_time_bins_index_I = find(position_time_bin_in_state);
    sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I = sum(Xnum_I, 2) ./ occ_I(:,1) ./ dt; %integral
Lint_I = Lint_I ./ sum(Lint_I);
end