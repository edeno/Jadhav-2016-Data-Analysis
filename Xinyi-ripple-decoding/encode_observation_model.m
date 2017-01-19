function [mark_spike_times, ...
    linear_distance_bins, ...
    linear_distance_bin_size, ...
    estimated_rate_all, ...
    mark_spike_by_tetrode, ...
    mark_spike_number_by_tetrode, ...
    marks, ...
    spike_times_to_linear_distance_time_bins_index_I, ...
    mark_smoothing_standard_deviation, ...
    place_field_estimator, ....
    linear_distance_occupancy, ...
    estimated_rate_by_tetrode ...
    ] = encode_observation_model(mark_spike_time0, marks, linear_distance, linear_distance_time, state_index)
% Whether the replay activity reflects spiking from inbound or outbound
% movements. Relate distance to channel maximums
num_linear_distance_bins = 61;
linear_distance_bins = linspace(min(linear_distance), max(linear_distance), num_linear_distance_bins);
linear_distance_bin_size = linear_distance_bins(2) - linear_distance_bins(1);
%% prepare kernel density model
dt = linear_distance_time(2) - linear_distance_time(1);
mark_smoothing_standard_deviation = 20;
num_discrete_states = length(state_index);
%% encode the kernel density model per tetrode
num_tetrodes = length(marks);
spike_times_to_linear_distance_time_bins_index_by_tetrode = cell(num_tetrodes, 1);

for tetrode_ind = 1:num_tetrodes,
    [spike_times_to_linear_distance_time_bins_index_by_tetrode{tetrode_ind}] = ...
        kernel_density_model(mark_spike_time0{tetrode_ind}, linear_distance_time);
end

spike_times_to_linear_distance_time_bins_index = cat(1, spike_times_to_linear_distance_time_bins_index_by_tetrode{:});
%% bookkeeping code: which spike comes which tetrode
tetrode_labels = cellfun(@(t, group) group * ones(size(t)), mark_spike_time0, num2cell(1:num_tetrodes)', 'uniformOutput', false);
tetrode_labels = cat(1, tetrode_labels{:});
[mark_spike_times, timeInd] = sort(cat(1, mark_spike_time0{:}));
spike_times_to_linear_distance_time_bins_index = spike_times_to_linear_distance_time_bins_index(timeInd, :);
tetrode_labels = tetrode_labels(timeInd);

mark_spike_by_tetrode = false(length(mark_spike_times), num_tetrodes);

for tetrode_ind = 1:num_tetrodes,
    mark_spike_by_tetrode(:, tetrode_ind) = (tetrode_labels == tetrode_ind);
end

mark_spike_number_by_tetrode = mark_spike_by_tetrode .* cumsum(mark_spike_by_tetrode, 1); %row: time point; column: index of spike per tetrode
%% captial LAMBDA (joint mark intensity function) conditioned on I=1 and I=0
estimated_rate_all = cell(num_discrete_states, 1);
linear_distance_occupancy = cell(num_discrete_states, 1);

for state_number = 1:num_discrete_states,
    [estimated_rate_all{state_number}, linear_distance_occupancy{state_number}] = condition_joint_mark_intensity_on_discrete_state(linear_distance', ...
        spike_times_to_linear_distance_time_bins_index, state_index{state_number}, ...
        linear_distance_bin_size, linear_distance_bins, dt);
end

% encode per tetrode, conditioning on I=1 and I=0
spike_times_to_linear_distance_time_bins_index_I = cell(num_tetrodes, num_discrete_states);
place_field_estimator = cell(num_tetrodes, num_discrete_states);
estimated_rate_by_tetrode = cell(num_tetrodes, num_discrete_states);

for tetrode_ind = 1:num_tetrodes,
    for state_number = 1:num_discrete_states,
        [spike_times_to_linear_distance_time_bins_index_I{tetrode_ind, state_number}, ...
            place_field_estimator{tetrode_ind, state_number}, ...
            estimated_rate_by_tetrode{tetrode_ind, state_number}] = encode_per_tetrode( ...
            spike_times_to_linear_distance_time_bins_index_by_tetrode{tetrode_ind}, ...
            state_index{state_number},  linear_distance_occupancy{state_number}, dt, ...
            linear_distance_bins, linear_distance', linear_distance_bin_size);
    end
end

save('computed_var.mat');
end