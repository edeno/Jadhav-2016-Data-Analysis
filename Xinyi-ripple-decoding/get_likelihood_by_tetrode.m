function [likelihood] = get_likelihood_by_tetrode(tet_sum, marks, ...
    mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, ...
    position_occupancy, estimated_rate_by_tetrode, dt, smker)
mark_spike_axis = ones(1, length(mark_spikes_to_linear_position_time_bins_index));
gaussian_kernel_place_mark_estimator = prod(normpdf(marks(tet_sum(mark_spike_axis), :), ...
    marks(mark_spikes_to_linear_position_time_bins_index, :), smker), 2);
joint_mark_intensity = gaussian_kernel_position_estimator * gaussian_kernel_place_mark_estimator ./ position_occupancy ./ dt;
likelihood = joint_mark_intensity .* dt .* exp(-estimated_rate_by_tetrode .* dt);
likelihood = normalize_distribution(likelihood);
end
