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
    spike_r = zeros(num_tetrodes, numSteps);
    
    %P(x0|I);
    Px_I{1} = exp(-stateV.^2  ./ (2 * (2 * stateV_delta)^2));
    Px_I{1} = Px_I{1} ./ sum(Px_I{1});
    Px_I{2} = max(Px_I{1}) * ones(1, stateV_length) - Px_I{1};
    Px_I{2} = Px_I{2} ./ sum(Px_I{2});
    
    %P(x0)=P(x0|I)P(I);
    postx{1} = 0.25 * Px_I{1}';
    postx{2} = 0.25 * Px_I{2}';
    postx{3} = 0.25 * Px_I{2}';
    postx{4} = 0.25 * Px_I{1}';
    pI0_vec = zeros(numSteps, 1);
    pI1_vec = zeros(numSteps, 1);
    pI2_vec = zeros(numSteps, 1);
    pI3_vec = zeros(numSteps, 1);
    
    %state transition
    stateM{1} = stateM_I_normalized_gaussian{1};
    stateM{2} = stateM_I_normalized_gaussian{2};
    stateM{3} = stateM_I_normalized_gaussian{2};
    stateM{4} = stateM_I_normalized_gaussian{1};
    
    for step_ind = 1:numSteps
        is_spike_at_time_t = find(mark_spike_times == position_time_stamps_binned(spike_tim(step_ind)));
        
        for state_ind = 1:length(postx),
            one_step_prediction_density{state_ind} = stateM{state_ind} * postx{state_ind};
        end
        
        if isempty(is_spike_at_time_t) %if no spike occurs at time t
            %% Is this supposed to happen? The labels don't seem right
            L{1} = exp(-Lint_I_Lambda{1} .* dt);
            L{2} = exp(-Lint_I_Lambda{1} .* dt);
            L{3} = exp(-Lint_I_Lambda{2} .* dt);
            L{4} = exp(-Lint_I_Lambda{2} .* dt);
            
        else %if spikes
            
            L_out = decode_per_state(1, stateV_length, is_spike_at_time_t, ...
                tet_ind, tet_sum, markAll, procInd1_I, procInd1_Ia, Xnum_I, occ_I_Lambda, Lint_I, dt, smker);
            L_in = decode_per_state(2, stateV_length, is_spike_at_time_t, ...
                tet_ind, tet_sum, markAll, procInd1_I, procInd1_Ia, Xnum_I, occ_I_Lambda, Lint_I, dt, smker);
            
            L{1} = L_out;
            L{2} = L_out;
            L{3} = L_in;
            L{4} = L_in;
        end
        
        totnorm = sum(one_step_prediction_density{1} .* L{1}) + ...
            sum(one_step_prediction_density{2} .* L{2}) + ...
            sum(one_step_prediction_density{3} .* L{3}) + ...
            sum(one_step_prediction_density{4} .* L{4});
        
        for state_ind = 1:length(postx),
            postx{state_ind} = one_step_prediction_density{state_ind} .* L{state_ind} ./ totnorm;
        end
        
        pI0_vec(step_ind) = sum(postx{1});
        pI1_vec(step_ind) = sum(postx{2});
        pI2_vec(step_ind) = sum(postx{3});
        pI3_vec(step_ind) = sum(postx{4});
    end
    
    summary_statistic{ripple_number} = [pI0_vec pI1_vec pI2_vec pI3_vec];
end
end