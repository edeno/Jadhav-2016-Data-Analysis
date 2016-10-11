function [procInd1_Ia, procInd1_I, Xnum_I, Lint_I] = encode_per_tetrode( ...
    procInd1, is_state,  occ_I, dt, linear_distance_bins, xtrain, sxker)

procInd1_Ia = procInd1(ismember(procInd1, is_state));
procInd1_I = find(ismember(procInd1, is_state));
Xnum_I = normpdf(linear_distance_bins' * ones(1, length(procInd1_Ia)), ...
    ones(length(linear_distance_bins), 1) * xtrain(procInd1_Ia), ...
    sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I = sum(Xnum_I, 2) ./ occ_I(:,1) ./ dt; %integral
Lint_I = Lint_I ./ sum(Lint_I);
end