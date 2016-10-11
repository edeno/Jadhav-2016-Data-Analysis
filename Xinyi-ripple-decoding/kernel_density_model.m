function [markAll, spike_times, mark0, procInd1] = kernel_density_model(animal, day, tetrode_number, ...
    linear_position_time)
mark_data_file = load(get_mark_filename('bond', day, tetrode_number));
params = mark_data_file.filedata.params;
ind = find(params(:, 1) / 1E4 >= linear_position_time(1) & ...
    params(:, 1) / 1E4 <= linear_position_time(end));
spike_times = params(ind, 1);
mark0 = params(ind, 2:5); % tetrode wire maxes
spike_time_in_seconds = spike_times / 1E4;
[~, procInd1] = histc(spike_time_in_seconds, linear_position_time);
[~, rawInd0] = histc(spike_time_in_seconds, spike_time_in_seconds);
markAll(:, 1) = procInd1;
markAll(:, 2:5) = mark0(rawInd0(rawInd0 ~= 0), :);
end

function [filename] = get_mark_filename(animal, day, tetrode_number)
    filename = sprintf('bond_data/%s%02d-%02d_params.mat', animal, day, tetrode_number);
end