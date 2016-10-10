function [l2] = decode_per_tetrode(tet_sum, markAll, procInd1_I, procInd1_Ia, Xnum_I, occ_Indicator, Lint_I, dt, smker)
new_axis = ones(1, length(procInd1_Ia));
l0 = normpdf(markAll(tet_sum, 2) * new_axis, markAll(procInd1_I, 2)', smker) .* ...
    normpdf(markAll(tet_sum, 3) * new_axis, markAll(procInd1_I, 3)', smker) .* ...
    normpdf(markAll(tet_sum, 4) * new_axis, markAll(procInd1_I, 4)', smker) .* ...
    normpdf(markAll(tet_sum, 5) * new_axis, markAll(procInd1_I, 5)', smker);
l1 = Xnum_I * l0' ./ occ_Indicator(:, 1) ./ dt;
l2 = l1 .* dt .* exp(-Lint_I .* dt);
l2 = l2 ./ sum(l2);
end
