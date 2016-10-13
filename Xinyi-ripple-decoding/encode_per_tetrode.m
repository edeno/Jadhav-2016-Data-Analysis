function [mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, estimated_rate] = encode_per_tetrode( ...
    mark_spikes_to_linear_position_time_bins_index, state_index,  position_occupancy, dt, linear_distance_bins, linear_distance, sxker)

num_linear_distance_bins = length(linear_distance_bins);
distance_bins_axis = ones(num_linear_distance_bins, 1);

position_time_bins_in_state = ismember(mark_spikes_to_linear_position_time_bins_index, state_index);
mark_spikes_to_linear_position_time_bins_index_Ia = mark_spikes_to_linear_position_time_bins_index(position_time_bins_in_state);

num_spikes = length(mark_spikes_to_linear_position_time_bins_index_Ia);
spikes_axis = ones(1, num_spikes);

mark_spikes_to_linear_position_time_bins_index = find(position_time_bins_in_state);

gaussian_kernel_position_estimator = normpdf(linear_distance_bins' * spikes_axis, ...
     distance_bins_axis * linear_distance(mark_spikes_to_linear_position_time_bins_index_Ia), ...
    sxker);

estimated_rate = sum(gaussian_kernel_position_estimator, 2) ./ position_occupancy ./ dt; %integral
estimated_rate = normalize_distirbution(estimated_rate);
end