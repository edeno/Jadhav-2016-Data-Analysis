function [summary_statistic] = decode_state(ripple_index, ...
    position_time_stamps_binned, ...
    mark_spike_times, ...
    stateV, ...
    stateV_delta, ...
    stateM_I_normalized_gaussian_outbound, ...
    stateM_I_normalized_gaussian_inbound, ...
    Lint_I_Lambda_outbound, ...
    Lint_I_Lambda_inbound, ...
    tet_ind, ...
    tet_sum, ...
    markAll, ...
    procInd1_Ia_out, ...
    procInd1_I_out, ...
    smker, ...
    Xnum_I_out, ....
    occ_Indicator_outbound, ...
    Lint_I_out, ...
    procInd1_Ia_in, ...
    procInd1_I_in, ...
    Xnum_I_in, ...
    occ_I_Lambda_inbound, ...
    Lint_I_in ...
    )

num_tetrodes = size(tet_ind, 2);
mark_spike_times = round(mark_spike_times / 10);

%only decode replay when the running speed < 4cm/sec
dt = 1 / 33.4;
stateV_length = length(stateV);

for ripple_number = 1:length(ripple_index)
    
    spike_tim = ripple_index(ripple_number, 1):ripple_index(ripple_number, 2);
    numSteps = length(spike_tim);
    spike_r = zeros(num_tetrodes, numSteps);

    %P(x0|I);
    Px_I_out = exp(-stateV.^2  ./ (2 * (2 * stateV_delta)^2));
    Px_I_out = Px_I_out ./ sum(Px_I_out);
    Px_I_in = max(Px_I_out) * ones(1, stateV_length) - Px_I_out;
    Px_I_in = Px_I_in ./ sum(Px_I_in);
    
    %P(x0)=P(x0|I)P(I);
    postx_I0 = 0.25 * Px_I_out';
    postx_I1 = 0.25 * Px_I_in';
    postx_I2 = 0.25 * Px_I_in';
    postx_I3 = 0.25 * Px_I_out';
    pI0_vec = zeros(numSteps, 1);
    pI1_vec = zeros(numSteps, 1);
    pI2_vec = zeros(numSteps, 1);
    pI3_vec = zeros(numSteps, 1);
    
    %state transition
    stateM_Indicator_outbound = stateM_I_normalized_gaussian_outbound;
    stateM_Indicator_inbound = stateM_I_normalized_gaussian_inbound;
    stateM_I0 = stateM_Indicator_outbound;
    stateM_I1 = stateM_Indicator_inbound;
    stateM_I2 = stateM_Indicator_inbound;
    stateM_I3 = stateM_Indicator_outbound;
    
    for step_ind = 1:numSteps       
        aa = find(mark_spike_times == position_time_stamps_binned(spike_tim(step_ind)));
        
        onestep_I0 = stateM_I0 * postx_I0;
        onestep_I1 = stateM_I1 * postx_I1;
        onestep_I2 = stateM_I2 * postx_I2;
        onestep_I3 = stateM_I3 * postx_I3;
        
        if isempty(aa) %if no spike occurs at time t
            L_I0 = exp(-Lint_I_Lambda_outbound .* dt);
            L_I1 = exp(-Lint_I_Lambda_outbound .* dt);
            L_I2 = exp(-Lint_I_Lambda_inbound .* dt);
            L_I3 = exp(-Lint_I_Lambda_inbound .* dt);
            
        else %if spikes
            
            l_out = zeros(stateV_length, length(aa));
            for j=1:length(aa)
                tetrode_ind = find(tet_ind(aa(j), :));
                spike_r(tetrode_ind, step_ind) = 1;
                l_out(:, j) = decode_per_tetrode(tet_sum(aa(j), tetrode_ind), ...
                    markAll{tetrode_ind}, procInd1_I_out{tetrode_ind}, procInd1_Ia_out{tetrode_ind}, ...
                    Xnum_I_out{tetrode_ind}, occ_Indicator_outbound, ...
                    Lint_I_out{tetrode_ind}, dt, smker);
            end
            L_out = prod(l_out,2);
            L_out = L_out ./ sum(L_out);
            
            l_in = zeros(stateV_length, length(aa));
            for j=1:length(aa)
                tetrode_ind = find(tet_ind(aa(j), :));
                spike_r(tetrode_ind, step_ind) = 1;
                l_in(:, j) = decode_per_tetrode(tet_sum(aa(j), tetrode_ind), ...
                    markAll{tetrode_ind}, procInd1_I_in{tetrode_ind}, procInd1_Ia_in{tetrode_ind}, ...
                    Xnum_I_in{tetrode_ind}, occ_I_Lambda_inbound, ...
                    Lint_I_in{tetrode_ind}, dt, smker);
            end
            L_in = prod(l_in, 2);
            L_in = L_in ./ sum(L_in);
            
            L_I0 = L_out;
            L_I1 = L_out;
            L_I2 = L_in;
            L_I3 = L_in;
        end
        
        totnorm = sum(onestep_I0 .* L_I0) + ...
            sum(onestep_I1 .* L_I1) + ...
            sum(onestep_I2 .* L_I2) + ...
            sum(onestep_I3 .* L_I3);
        postx_I0 = onestep_I0 .* L_I0 ./ totnorm;
        postx_I1 = onestep_I1 .* L_I1 ./ totnorm;
        postx_I2 = onestep_I2 .* L_I2 ./ totnorm;
        postx_I3 = onestep_I3 .* L_I3 ./ totnorm;
        
        pI0_vec(step_ind) = sum(postx_I0);
        pI1_vec(step_ind) = sum(postx_I1);
        pI2_vec(step_ind) = sum(postx_I2);
        pI3_vec(step_ind) = sum(postx_I3);
    end
    
    summary_statistic{ripple_number} = [pI0_vec pI1_vec pI2_vec pI3_vec];
end
end