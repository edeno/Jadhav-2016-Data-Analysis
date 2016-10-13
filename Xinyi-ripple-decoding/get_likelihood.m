function [likelihood] = get_likelihood(spike_inds, estimated_rate_all, dt, ...
    num_linear_distance_bins, mark_spike_by_tetrode, mark_spike_number_by_tetrode, ...
    marks, mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, ...
    position_occupancy, estimated_rate_by_tetrode, mark_smoothing_standard_deviation)
if isempty(spike_inds) %if no spike occurs at time step
    likelihood(:, 1) = exp(-estimated_rate_all{1} .* dt);
    likelihood(:, 2) = exp(-estimated_rate_all{1} .* dt);
    likelihood(:, 3) = exp(-estimated_rate_all{2} .* dt);
    likelihood(:, 4) = exp(-estimated_rate_all{2} .* dt);
else %if spikes
    likelihood_outbound = get_likelihood_by_state(1, num_linear_distance_bins, spike_inds, ...
        mark_spike_by_tetrode, mark_spike_number_by_tetrode, marks, mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, position_occupancy, estimated_rate_by_tetrode, dt, mark_smoothing_standard_deviation);
    likelihood_inbound = get_likelihood_by_state(2, num_linear_distance_bins, spike_inds, ...
        mark_spike_by_tetrode, mark_spike_number_by_tetrode, marks, mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, position_occupancy, estimated_rate_by_tetrode, dt, mark_smoothing_standard_deviation);
    
    likelihood(:, 1) = likelihood_outbound;
    likelihood(:, 2) = likelihood_outbound;
    likelihood(:, 3) = likelihood_inbound;
    likelihood(:, 4) = likelihood_inbound;
end
end