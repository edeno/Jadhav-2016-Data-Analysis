function [estimated_rate, position_occupancy] = condition_joint_mark_intensity_on_discrete_state(xtrain, ...
    mark_spikes_to_linear_position_time_bins_index, state_index, sxker, mark_bins, linear_distance_bins, dt)
num_mark_bins = length(mark_bins);
num_state = length(state_index);
linear_distance_bins_axis = ones(length(linear_distance_bins), 1);
state_axis = ones(1, num_state);

mark_spikes_to_linear_position_time_bins_index = mark_spikes_to_linear_position_time_bins_index(ismember(mark_spikes_to_linear_position_time_bins_index, state_index));
num_spikes_axis = ones(1, length(mark_spikes_to_linear_position_time_bins_index));
position_occupancy = normpdf(linear_distance_bins(state_axis, :)', ...
    xtrain(linear_distance_bins_axis, state_index), sxker) * ...
    ones(num_state, num_mark_bins);
gaussian_kernel_position_estimator_I = normpdf(linear_distance_bins(num_spikes_axis, :)', ...
     xtrain(linear_distance_bins_axis, mark_spikes_to_linear_position_time_bins_index), ...
    sxker);

estimated_rate = sum(gaussian_kernel_position_estimator_I, 2) ./ position_occupancy(:, 1) ./ dt; %integral
estimated_rate = estimated_rate ./ sum(estimated_rate);
end