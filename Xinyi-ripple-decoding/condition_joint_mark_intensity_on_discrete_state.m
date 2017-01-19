function [estimated_rate, linear_distance_occupancy] = condition_joint_mark_intensity_on_discrete_state(linear_distance, ...
    spike_times_to_linear_distance_time_bins_index, state_index, linear_distance_bin_size, linear_distance_bins, dt)
num_state = length(state_index);
linear_distance_bins_axis = ones(length(linear_distance_bins), 1);
state_axis = ones(1, num_state);

spike_times_to_linear_distance_time_bins_index = spike_times_to_linear_distance_time_bins_index(ismember(spike_times_to_linear_distance_time_bins_index, state_index));
num_spikes_axis = ones(1, length(spike_times_to_linear_distance_time_bins_index));

linear_distance_occupancy = sum(normpdf(linear_distance_bins(state_axis, :)', linear_distance(linear_distance_bins_axis, state_index), linear_distance_bin_size), 2);
linear_distance_at_spike = linear_distance(linear_distance_bins_axis, spike_times_to_linear_distance_time_bins_index);
gaussian_kernel_distance_estimator = normpdf(linear_distance_bins(num_spikes_axis, :)', ...
     linear_distance_at_spike, linear_distance_bin_size);

estimated_rate = sum(gaussian_kernel_distance_estimator, 2) ./ linear_distance_occupancy ./ dt; %integral
estimated_rate = normalize_distribution(estimated_rate);
end