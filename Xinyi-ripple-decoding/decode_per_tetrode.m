function [l2] = decode_per_tetrode(tet_sum, marks, procInd1_I, Xnum_I, occ_Indicator, Lint_I, dt, smker)
new_axis = ones(1, length(procInd1_I));
l0 = normpdf(marks(tet_sum, 1) * new_axis, marks(procInd1_I, 1)', smker) .* ...
    normpdf(marks(tet_sum, 2) * new_axis, marks(procInd1_I, 2)', smker) .* ...
    normpdf(marks(tet_sum, 3) * new_axis, marks(procInd1_I, 3)', smker) .* ...
    normpdf(marks(tet_sum, 4) * new_axis, marks(procInd1_I, 4)', smker);
l1 = Xnum_I * l0' ./ occ_Indicator(:, 1) ./ dt;
l2 = l1 .* dt .* exp(-Lint_I .* dt);
l2 = l2 ./ sum(l2);
end
