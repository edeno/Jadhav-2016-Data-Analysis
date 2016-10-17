function [ripple_time_extent, position_time_milliseconds, dt] = get_ripple_index(pos, ripplescons, spikes, tetrode_index, neuron_index)
%% calculate ripple starting and end times
position_time = pos.data(:, 1); %time stamps for animal's trajectory in seconds
position_time_milliseconds = to_milliseconds(position_time(1)):to_milliseconds(position_time(end)); %position time stamps in milliseconds

[ripple_time_extent] = get_ripple_extent(ripplescons, ...
    position_time_milliseconds);

[num_spikes_per_ripple] = get_number_spikes_per_ripple(tetrode_index, ...
    neuron_index, spikes, position_time_milliseconds, ripple_time_extent);


[running_speed_at_ripple] = get_running_speed_at_ripple(pos, ripple_time_extent, ...
    position_time, position_time_milliseconds);

ripple_time_extent = ripple_time_extent(running_speed_at_ripple < 4 & num_spikes_per_ripple > 0, :);
dt = 1 / (1000 * (position_time(2) - position_time(1)));
end

function [x] = to_milliseconds(x)
x = round(x * 1000);
end

function [ripple_time_extent] = get_ripple_extent(ripplescons, ...
    position_time_milliseconds)
ripple_start_time = to_milliseconds(ripplescons{1}.starttime);
ripple_end_time = to_milliseconds(ripplescons{1}.endtime);
traj_Ind = find(ripplescons{1}.maxthresh > 4);
ripple_start_time = ripple_start_time(traj_Ind);
ripple_end_time = ripple_end_time(traj_Ind);
ripple_time_extent = [ripple_start_time - position_time_milliseconds(1) - 1, ...
    ripple_end_time - position_time_milliseconds(1) - 1];
end

function [running_speed_at_ripple] = get_running_speed_at_ripple(pos, ...
    ripple_time_extent, position_time_stamps, position_time_milliseconds)
running_speed = pos.data(:, 5);
running_speed_at_ripple = nan(size(ripple_time_extent, 1), 1);
for ripple_number = 1:size(ripple_time_extent, 1)
    ripple_ind = find(position_time_stamps * 1000 > position_time_milliseconds(ripple_time_extent(ripple_number, 1)) & ...
        position_time_stamps * 1000 < position_time_milliseconds(ripple_time_extent(ripple_number, 2)));
    running_speed_at_ripple(ripple_number) = running_speed(ripple_ind(1), 1);
end
end

function [num_spikes_per_ripple] = get_number_spikes_per_ripple(tetrode_index, ...
    neuron_index, spikes, position_time_milliseconds, ripple_time_extent)
for neuron_ind = 1:size(tetrode_index, 2)
    spike_times_milliseconds = to_milliseconds(spikes{tetrode_index(neuron_ind)}{neuron_index(neuron_ind)}.data(:,1));
    is_spike(neuron_ind, :) = ismember(position_time_milliseconds, spike_times_milliseconds);
end

num_spikes_per_ripple = nan(size(ripple_time_extent, 1), 1);
for ripple_number = 1:size(ripple_time_extent, 1)
    is_ripple_spike = is_spike(:, ripple_time_extent(ripple_number, 1):ripple_time_extent(ripple_number, 2));
    num_spikes_per_ripple(ripple_number) = sum(is_ripple_spike(:));
end

end