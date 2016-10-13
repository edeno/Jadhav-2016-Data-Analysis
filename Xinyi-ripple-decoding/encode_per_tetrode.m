function [mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, estimated_rate] = encode_per_tetrode( ...
    mark_spikes_to_linear_position_time_bins_index, state_index,  position_occupancy, dt, linear_distance_bins, xtrain, sxker)

num_linear_distance_bins = length(linear_distance_bins);
position_time_bins_in_state = ismember(mark_spikes_to_linear_position_time_bins_index, state_index);
mark_spikes_to_linear_position_time_bins_index_Ia = mark_spikes_to_linear_position_time_bins_index(position_time_bins_in_state);
mark_spikes_to_linear_position_time_bins_index = find(position_time_bins_in_state);
gaussian_kernel_position_estimator = normpdf(linear_distance_bins' * ones(1, length(mark_spikes_to_linear_position_time_bins_index_Ia)), ...
    ones(num_linear_distance_bins, 1) * xtrain(mark_spikes_to_linear_position_time_bins_index_Ia), ...
    sxker);
estimated_rate = sum(gaussian_kernel_position_estimator, 2) ./ position_occupancy ./ dt; %integral
estimated_rate = estimated_rate ./ sum(estimated_rate);
end