function likelihood = get_likelihood_by_state(state_number, stateV_length, is_spike_at_time_t, tet_ind, tet_sum, marks, procInd1_I, Xnum_I, occ_I_Lambda, Lint_I, dt, smker)
l = zeros(stateV_length, length(is_spike_at_time_t));
for j = 1:length(is_spike_at_time_t)
    tetrode_ind = find(tet_ind(is_spike_at_time_t(j), :));
    l(:, j) = get_likelihood_by_tetrode(tet_sum(is_spike_at_time_t(j), tetrode_ind), ...
        marks{tetrode_ind}, procInd1_I{tetrode_ind, state_number}, ...
        Xnum_I{tetrode_ind, state_number}, occ_I_Lambda{state_number}, ...
        Lint_I{tetrode_ind, state_number}, dt, smker);
end
likelihood = prod(l, 2);
likelihood = likelihood ./ sum(likelihood);
end