function [l2] = get_likelihood_by_tetrode(tet_sum, marks, ...
    mark_spikes_to_linear_position_time_bins_index_I, gaussian_kernel_position_estimator_I, ...
    occ_Indicator, estimated_rate_by_tetrode, dt, smker)
mark_spike_axis = ones(1, length(mark_spikes_to_linear_position_time_bins_index_I));
l0 = prod(normpdf(marks(tet_sum(mark_spike_axis), :), marks(mark_spikes_to_linear_position_time_bins_index_I, :), smker), 2);
l1 = gaussian_kernel_position_estimator_I * l0 ./ occ_Indicator(:, 1) ./ dt;
l2 = l1 .* dt .* exp(-estimated_rate_by_tetrode .* dt);
l2 = l2 ./ sum(l2);
end
