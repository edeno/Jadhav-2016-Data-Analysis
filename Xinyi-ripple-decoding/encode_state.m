function [mark_spike_times, ...
    stateV, ...
    stateV_delta, ...
    stateM_I_normalized_gaussian, ...
    Lint_I_Lambda, ...
    tet_ind, ...
    tet_sum, ...
    marks, ...
    procInd1_I, ...
    smker, ...
    Xnum_I, ....
    occ_I_Lambda, ...
    Lint_I ...
    ] = encode_state(animal, day, linear_distance, linear_position_time, state_index, tetrode_number)
%% use Loren's linearization
num_linear_distance_bins = 61;
stateV = linspace(min(linear_distance), max(linear_distance), num_linear_distance_bins);
stateV_delta = stateV(2) - stateV(1);
%% empirical movement transition matrix conditioned on discrete state
num_discrete_states = length(state_index);
stateM_I_normalized_gaussian = cell(num_discrete_states, 1);
for state_number = 1:num_discrete_states,
    stateM_I_normalized_gaussian{state_number} = condition_empirical_movement_transition_matrix_on_state(stateV, linear_distance, state_index{state_number});
end

%% prepare kernel density model
linear_distance_bins = min(linear_distance):stateV_delta:max(linear_distance);
dt = linear_position_time(2) - linear_position_time(1);
xtrain = linear_distance';

sxker = stateV_delta;
mdel = 20;
smker = mdel;
%% encode the kernel density model per tetrode
num_tetrodes = length(tetrode_number);

mark_spike_time0 = cell(num_tetrodes, 1);
marks = cell(num_tetrodes, 1);

for tetrode_ind = 1:num_tetrodes,
    [mark_spike_time0{tetrode_ind}, marks{tetrode_ind}, ...
        linear_position_time);
end

mark0 = cat(1, mark0{:});
procInd1 = cat(1, procInd1_tet{:});
%% bookkeeping code: which spike comes which tetrode
group_labels = cellfun(@(t, group) group * ones(size(t)), mark_spike_time0, num2cell(1:num_tetrodes)', 'uniformOutput', false);
group_labels = cat(1, group_labels{:});
[mark_spike_times, timeInd] = sort(cat(1, mark_spike_time0{:}));
procInd1 = procInd1(timeInd, :);

tet_ind = false(length(mark_spike_times), num_tetrodes);

for tetrode_ind = 1:num_tetrodes,
    tet_ind(:, tetrode_ind) = (group_labels(timeInd) == tetrode_ind);
end

tet_sum = tet_ind .* cumsum(tet_ind,1); %row: time point; column: index of spike per tetrode

%% captial LAMBDA (joint mark intensity function) conditioned on I=1 and I=0
mark_bins = min(cat(1, marks{:})):mdel:max(cat(1, marks{:}));
Lint_I_Lambda = cell(num_discrete_states, 1);
occ_I_Lambda = cell(num_discrete_states, 1);

for state_number = 1:num_discrete_states,
    [Lint_I_Lambda{state_number}, occ_I_Lambda{state_number}] = condition_joint_mark_intensity_on_discrete_state(xtrain, procInd1, state_index{state_number}, sxker, mark_bins, linear_distance_bins, dt);
end

% encode per tetrode, conditioning on I=1 and I=0
procInd1_I = cell(num_tetrodes, num_discrete_states);
Xnum_I = cell(num_tetrodes, num_discrete_states);
Lint_I = cell(num_tetrodes, num_discrete_states);

for tetrode_ind = 1:num_tetrodes,
    for state_number = 1:num_discrete_states,
        [procInd1_I{tetrode_ind, state_number}, Xnum_I{tetrode_ind, state_number}, ...
            Lint_I{tetrode_ind, state_number}] = encode_per_tetrode( ...
            procInd1_tet{tetrode_ind}, state_index{state_number},  occ_I_Lambda{state_number}, dt, linear_distance_bins, xtrain, sxker);
    end
end

save('computed_var.mat');
end