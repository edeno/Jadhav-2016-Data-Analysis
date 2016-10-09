clear variables; clc;
expected_var = load('expected_var.mat');
computed_var = load('computed_var.mat');
tetrode_number = [1 2 4 5 7 10 11 12 13 14 17 18 19 20 22 23 27 29]; %tetrode index
names = {'markAll_t%d', 'procInd1_t%d_Ia_out', 'procInd1_t%d_I_out', ...
    'Xnum_t%d_I_out', 'Lint_t%d_I_out', 'procInd1_t%d_Ia_in', ...
    'procInd1_t%d_I_in', 'Xnum_t%d_I_in', 'Lint_t%d_I_in'};

for name_ind = 1:length(names),
    for n = tetrode_number,
        field_name = sprintf(names{name_ind}, n);
        if ~isequal(expected_var.(field_name), computed_var.(field_name)),
            fprintf('\n');
            fprintf(field_name);
        end
    end
end