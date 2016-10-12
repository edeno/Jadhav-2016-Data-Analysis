function [mark_spike_times, marks, mark_spikes_to_linear_position_time_bins_index] = kernel_density_model(animal, day, tetrode_number, ...
    linear_position_time)
mark_data_file = load(get_mark_filename('bond', day, tetrode_number));
params = mark_data_file.filedata.params;
mark_spike_times = params(:, 1);
mark_spike_time_in_seconds = mark_spike_times / 1E4;
ind = find(mark_spike_time_in_seconds >= linear_position_time(1) & ...
    mark_spike_time_in_seconds <= linear_position_time(end));

marks = params(ind, 2:5); % tetrode wire maxes
mark_spike_times = mark_spike_times(ind);
mark_spike_time_in_seconds = mark_spike_times / 1E4;

[~, mark_spikes_to_linear_position_time_bins_index] = histc(mark_spike_time_in_seconds, linear_position_time);
end

function [filename] = get_mark_filename(animal, day, tetrode_number)
    filename = sprintf('bond_data/%s%02d-%02d_params.mat', animal, day, tetrode_number);
end