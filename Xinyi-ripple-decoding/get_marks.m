function [mark_spike_times, marks] = get_marks(animal, day, tetrode_number)
num_tetrodes = length(tetrode_number);
mark_spike_times = cell(num_tetrodes, 1);
marks = cell(num_tetrodes, 1);
for tetrode_ind = 1:num_tetrodes,
    [mark_spike_times{tetrode_ind}, marks{tetrode_ind}] = get_mark_data_by_tetrode(animal, day, tetrode_number(tetrode_ind));
end


end

function [mark_spike_times, marks] = get_mark_data_by_tetrode(animal, day, tetrode_number)
mark_data_file = load(get_mark_filename('bond', day, tetrode_number));
params = mark_data_file.filedata.params;
mark_spike_times = params(:, 1);
marks = params(:, 2:5); % tetrode wire maxes
end

function [filename] = get_mark_filename(animal, day, tetrode_number)
filename = sprintf('bond_data/%s%02d-%02d_params.mat', animal, day, tetrode_number);
end