function [procInd1_Ia_out, procInd1_Ia_in, procInd1_I_out, procInd1_I_in, ...
    Xnum_I_out, Xnum_I_in, Lint_I_out, Lint_I_in] = ...
    encode_per_tetrode(procInd1, ind_Indicator_outbound, ind_Indicator_inbound, ...
    occ_Indicator_outbound, occ_Indicator_inbound, dt, xs, xtrain, sxker)
procInd1_Ia_out = procInd1(ismember(procInd1, ind_Indicator_outbound));
procInd1_I_out = find(ismember(procInd1, ind_Indicator_outbound));
Xnum_I_out = normpdf(xs' * ones(1, length(procInd1_Ia_out)), ones(length(xs), 1) * xtrain(procInd1_Ia_out), sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I_out = sum(Xnum_I_out, 2) ./ occ_Indicator_outbound(:,1) ./ dt; %integral
Lint_I_out = Lint_I_out ./ sum(Lint_I_out);
procInd1_Ia_in = procInd1(ismember(procInd1, ind_Indicator_inbound));
procInd1_I_in = find(ismember(procInd1, ind_Indicator_inbound));
Xnum_I_in = normpdf(xs' * ones(1,length(procInd1_Ia_in)), ones(length(xs),1) * xtrain(procInd1_Ia_in), sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I_in = sum(Xnum_I_in, 2) ./ occ_Indicator_inbound(:, 1) ./ dt; %integral
Lint_I_in = Lint_I_in ./ sum(Lint_I_in);
end