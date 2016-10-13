function [l2] = get_likelihood_by_tetrode(tet_sum, marks, ...
    mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, ...
    position_occupancy, estimated_rate_by_tetrode, dt, smker)
mark_spike_axis = ones(1, length(mark_spikes_to_linear_position_time_bins_index));
l0 = prod(normpdf(marks(tet_sum(mark_spike_axis), :), marks(mark_spikes_to_linear_position_time_bins_index, :), smker), 2);
l1 = gaussian_kernel_position_estimator * l0 ./ position_occupancy ./ dt;
l2 = l1 .* dt .* exp(-estimated_rate_by_tetrode .* dt);
l2 = normalize_distribution(l2);
end
