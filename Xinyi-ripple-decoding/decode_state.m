function [summary_statistic] = decode_state(ripple_index, ...
    position_time_stamps_binned, ...
    mark_spike_times, ...
    linear_distance_bins, ...
    linear_distance_bin_size, ...
    empirical_movement_transition_matrix, ...
    estimated_rate_all, ...
    mark_spike_by_tetrode, ...
    mark_spike_number_by_tetrode, ...
    marks, ...
    mark_spikes_to_linear_position_time_bins_index, ...
    smker, ...
    gaussian_kernel_position_estimator, ....
    position_occupancy, ...
    estimated_rate_by_tetrode ...
    )

mark_spike_times = round(mark_spike_times / 10);

dt = 1 / 33.4;
num_linear_distance_bins = length(linear_distance_bins);

for ripple_number = 1:length(ripple_index)
    
    ripple_time = ripple_index(ripple_number, 1):ripple_index(ripple_number, 2);
    numSteps = length(ripple_time);
    
    %P(x0|I);
    Px_I{1} = exp(-linear_distance_bins.^2  ./ (2 * (2 * linear_distance_bin_size)^2));
    Px_I{1} = Px_I{1} ./ sum(Px_I{1});
    Px_I{2} = max(Px_I{1}) * ones(1, num_linear_distance_bins) - Px_I{1};
    Px_I{2} = Px_I{2} ./ sum(Px_I{2});
    
    %P(x0)=P(x0|I)P(I);
    
    posterior_density(:, 1) = 0.25 * Px_I{1}';
    posterior_density(:, 2) = 0.25 * Px_I{2}';
    posterior_density(:, 3) = 0.25 * Px_I{2}';
    posterior_density(:, 4) = 0.25 * Px_I{1}';
    decision_state_probability = zeros(numSteps, 4);
    
    state_transition_model{1} = empirical_movement_transition_matrix{1}; % outbound forward
    state_transition_model{2} = empirical_movement_transition_matrix{2}; % outbound reverse
    state_transition_model{3} = empirical_movement_transition_matrix{2}; % inbound forward
    state_transition_model{4} = empirical_movement_transition_matrix{1}; % inbound reverse
    
    for time_step_ind = 1:numSteps
        spike_inds = find(mark_spike_times == position_time_stamps_binned(ripple_time(time_step_ind)));
        
        for decision_state_ind = 1:size(posterior_density, 2),
            one_step_prediction_density(:, decision_state_ind) = state_transition_model{decision_state_ind} * posterior_density(:, decision_state_ind);
        end
        
        if isempty(spike_inds) %if no spike occurs at time step
            likelihood(:, 1) = exp(-estimated_rate_all{1} .* dt);
            likelihood(:, 2) = exp(-estimated_rate_all{1} .* dt);
            likelihood(:, 3) = exp(-estimated_rate_all{2} .* dt);
            likelihood(:, 4) = exp(-estimated_rate_all{2} .* dt);
        else %if spikes   
            likelihood_outbound = get_likelihood_by_state(1, num_linear_distance_bins, spike_inds, ...
                mark_spike_by_tetrode, mark_spike_number_by_tetrode, marks, mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, position_occupancy, estimated_rate_by_tetrode, dt, smker);
            likelihood_inbound = get_likelihood_by_state(2, num_linear_distance_bins, spike_inds, ...
                mark_spike_by_tetrode, mark_spike_number_by_tetrode, marks, mark_spikes_to_linear_position_time_bins_index, gaussian_kernel_position_estimator, position_occupancy, estimated_rate_by_tetrode, dt, smker);
            
            likelihood(:, 1) = likelihood_outbound;
            likelihood(:, 2) = likelihood_outbound;
            likelihood(:, 3) = likelihood_inbound;
            likelihood(:, 4) = likelihood_inbound;
        end
        
        total_norm = sum(one_step_prediction_density(:) .* likelihood(:));
        posterior_density = one_step_prediction_density .* likelihood / total_norm;
        decision_state_probability(time_step_ind, :) = sum(posterior_density);

    end
    
    summary_statistic{ripple_number} = decision_state_probability;
end
end