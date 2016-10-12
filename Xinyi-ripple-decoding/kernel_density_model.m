function [markAll, spike_times, marks, procInd1] = kernel_density_model(animal, day, tetrode_number, ...
    linear_position_time)
mark_data_file = load(get_mark_filename('bond', day, tetrode_number));
params = mark_data_file.filedata.params;
ind = find(params(:, 1) / 1E4 >= linear_position_time(1) & ...
    params(:, 1) / 1E4 <= linear_position_time(end));
spike_times = params(ind, 1);
marks = params(ind, 2:5); % tetrode wire maxes
spike_time_in_seconds = spike_times / 1E4;
[~, procInd1] = histc(spike_time_in_seconds, linear_position_time);
[~, rawInd0] = histc(spike_time_in_seconds, spike_time_in_seconds);
markAll = marks(rawInd0(rawInd0 ~= 0), :);
end

function [filename] = get_mark_filename(animal, day, tetrode_number)
    filename = sprintf('bond_data/%s%02d-%02d_params.mat', animal, day, tetrode_number);
end