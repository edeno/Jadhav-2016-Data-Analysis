function [spike_times_to_linear_distance_time_bins_index, place_field_estimator, estimated_rate] = encode_per_tetrode( ...
    spike_times_to_linear_distance_time_bins_index, state_index, linear_distance_occupancy, dt, linear_distance_bins, linear_distance, linear_distance_bin_size)

num_linear_distance_bins = length(linear_distance_bins);
linear_distance_bins_axis = ones(num_linear_distance_bins, 1);

linear_distance_time_bins_in_state = ismember(spike_times_to_linear_distance_time_bins_index, state_index);
spike_times_to_linear_distance_time_bins_index_at_state = spike_times_to_linear_distance_time_bins_index(linear_distance_time_bins_in_state);

num_spikes = length(spike_times_to_linear_distance_time_bins_index_at_state);
spikes_axis = ones(1, num_spikes);

spike_times_to_linear_distance_time_bins_index = find(linear_distance_time_bins_in_state);

place_field_estimator = normpdf(linear_distance_bins' * spikes_axis, ...
    linear_distance_bins_axis * linear_distance(spike_times_to_linear_distance_time_bins_index_at_state), ...
    linear_distance_bin_size);

estimated_rate = sum(place_field_estimator, 2) ./ linear_distance_occupancy ./ dt; %integral
estimated_rate = normalize_distribution(estimated_rate);
end