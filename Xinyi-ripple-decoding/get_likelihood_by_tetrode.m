function [likelihood] = get_likelihood_by_tetrode(spike_number, marks, ...
    spike_times_to_linear_distance_time_bins_index, place_field_estimator, ...
    linear_distance_occupancy, estimated_rate_by_tetrode, dt, smker)
mark_spike_axis = ones(1, length(spike_times_to_linear_distance_time_bins_index));
mark_space_estimator = prod(normpdf(marks(spike_number(mark_spike_axis), :), ...
    marks(spike_times_to_linear_distance_time_bins_index, :), smker), 2);
joint_mark_intensity = place_field_estimator * mark_space_estimator ./ linear_distance_occupancy ./ dt;
likelihood = joint_mark_intensity .* dt .* exp(-estimated_rate_by_tetrode .* dt);
likelihood = normalize_distribution(likelihood);
end
