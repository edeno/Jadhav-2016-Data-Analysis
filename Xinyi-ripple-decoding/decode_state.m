function [summary_statistic] = decode_state(ripple_index, ...
    position_time_stamps_binned, ...
    mark_spike_times, ...
    stateV, ...
    stateV_delta, ...
    stateM_I_normalized_gaussian, ...
    Lint_I_Lambda, ...
    tet_ind, ...
    tet_sum, ...
    markAll, ...
    procInd1_Ia, ...
    procInd1_I, ...
    smker, ...
    Xnum_I, ....
    occ_I_Lambda, ...
    Lint_I ...
    )

num_tetrodes = size(tet_ind, 2);
mark_spike_times = round(mark_spike_times / 10);

dt = 1 / 33.4;
stateV_length = length(stateV);

for ripple_number = 1:length(ripple_index)
    
    spike_tim = ripple_index(ripple_number, 1):ripple_index(ripple_number, 2);
    numSteps = length(spike_tim);
    
    %P(x0|I);
    Px_I{1} = exp(-stateV.^2  ./ (2 * (2 * stateV_delta)^2));
    Px_I{1} = Px_I{1} ./ sum(Px_I{1});
    Px_I{2} = max(Px_I{1}) * ones(1, stateV_length) - Px_I{1};
    Px_I{2} = Px_I{2} ./ sum(Px_I{2});
    
    %P(x0)=P(x0|I)P(I);
    
    posterior_density{1} = 0.25 * Px_I{1}';
    posterior_density{2} = 0.25 * Px_I{2}';
    posterior_density{3} = 0.25 * Px_I{2}';
    posterior_density{4} = 0.25 * Px_I{1}';
    decision_state_probability = zeros(numSteps, 4);
    
    state_transition_model{1} = stateM_I_normalized_gaussian{1}; % outbound forward
    state_transition_model{2} = stateM_I_normalized_gaussian{2}; % outbound reverse
    state_transition_model{3} = stateM_I_normalized_gaussian{2}; % inbound forward
    state_transition_model{4} = stateM_I_normalized_gaussian{1}; % inbound reverse
    
    for time_step_ind = 1:numSteps
        is_spike_at_time_t = find(mark_spike_times == position_time_stamps_binned(spike_tim(time_step_ind)));
        
        for decision_state_ind = 1:length(posterior_density),
            one_step_prediction_density(:, decision_state_ind) = state_transition_model{decision_state_ind} * posterior_density{decision_state_ind};
        end
        
        if isempty(is_spike_at_time_t) %if no spike occurs at time t
            %% Is this supposed to happen? The labels don't seem right
            likelihood(:, 1) = exp(-Lint_I_Lambda{1} .* dt);
            likelihood(:, 2) = exp(-Lint_I_Lambda{1} .* dt);
            likelihood(:, 3) = exp(-Lint_I_Lambda{2} .* dt);
            likelihood(:, 4) = exp(-Lint_I_Lambda{2} .* dt);
            
        else %if spikes
            
            likelihood_outbound = decode_per_state(1, stateV_length, is_spike_at_time_t, ...
                tet_ind, tet_sum, markAll, procInd1_I, procInd1_Ia, Xnum_I, occ_I_Lambda, Lint_I, dt, smker);
            likelihood_inbound = decode_per_state(2, stateV_length, is_spike_at_time_t, ...
                tet_ind, tet_sum, markAll, procInd1_I, procInd1_Ia, Xnum_I, occ_I_Lambda, Lint_I, dt, smker);
            
            likelihood(:, 1) = likelihood_outbound;
            likelihood(:, 2) = likelihood_outbound;
            likelihood(:, 3) = likelihood_inbound;
            likelihood(:, 4) = likelihood_inbound;
        end
        
        total_norm = sum(one_step_prediction_density(:) .* likelihood(:));
        
        for decision_state_ind = 1:length(posterior_density),
            posterior_density{decision_state_ind} = one_step_prediction_density(:, decision_state_ind) .* likelihood(:, decision_state_ind) ./ total_norm;
            decision_state_probability(time_step_ind, decision_state_ind) = sum(posterior_density{decision_state_ind});
        end

    end
    
    summary_statistic{ripple_number} = decision_state_probability;
end
end