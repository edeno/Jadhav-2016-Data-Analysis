function likelihood = decode_per_state(state_number, stateV_length, is_spike_at_time_t, tet_ind, tet_sum, markAll, procInd1_I, Xnum_I, occ_I_Lambda, Lint_I, dt, smker)
l = zeros(stateV_length, length(is_spike_at_time_t));
for j = 1:length(is_spike_at_time_t)
    tetrode_ind = find(tet_ind(is_spike_at_time_t(j), :));
    l(:, j) = decode_per_tetrode(tet_sum(is_spike_at_time_t(j), tetrode_ind), ...
        markAll{tetrode_ind}, procInd1_I{tetrode_ind, state_number}, ...
        Xnum_I{tetrode_ind, state_number}, occ_I_Lambda{state_number}, ...
        Lint_I{tetrode_ind, state_number}, dt, smker);
end
likelihood = prod(l, 2);
likelihood = likelihood ./ sum(likelihood);
end