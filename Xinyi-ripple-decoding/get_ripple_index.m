function [ripple_index, position_time_stamps_milliseconds] = get_ripple_index(pos, ripplescons, spikes, tetrode_index, neuron_index)
%% calculate ripple starting and end times
position_time_stamps = pos.data(:, 1); %time stamps for animal's trajectory in seconds
position_time_stamps_milliseconds = to_milliseconds(position_time_stamps(1)):to_milliseconds(position_time_stamps(end)); %position time stamps in milliseconds

ripple_start_time = to_milliseconds(ripplescons{1}.starttime);
ripple_end_time = to_milliseconds(ripplescons{1}.endtime);
traj_Ind = find(ripplescons{1}.maxthresh > 4);
ripple_start_time = ripple_start_time(traj_Ind);
ripple_end_time = ripple_end_time(traj_Ind);
ripple_index = [ripple_start_time - position_time_stamps_milliseconds(1) - 1, ...
    ripple_end_time - position_time_stamps_milliseconds(1) - 1]; %index for ripple segments

for neuron_ind = 1:size(tetrode_index, 2)
    spike_times = spikes{tetrode_index(neuron_ind)}{neuron_index(neuron_ind)}.data(:,1); %spiking times for tetrode j, cell i
    binned_spike_times = round(spike_times * 1000); %binning spiking times at 1 ms
    [sptrain2_list{neuron_ind}, ~] = ismember(position_time_stamps_binned, binned_spike_times); %sptrain2: spike train binned at 1 ms instead of 33.4ms (sptrain0)
end

% Only ripples with spikes
for k = 1:size(ripple_index, 1)
    spike_r = [];
    for neuron_ind = 1:size(tetrode_index, 2)
        sptrain2 = sptrain2_list{neuron_ind};
        spike_r = [spike_r; sptrain2(ripple_index(k, 1):ripple_index(k ,2))];
    end
    spike_r_all{k} = spike_r;
end

for k = 1:size(ripple_index, 1)
    num_spikes_per_ripple(k) = sum(spike_r_all{k}(:));
end

running_speed = pos.data(:,5);
% Only ripples with the rat running < 4 cm / sec
for ripple_number = 1:size(ripple_index, 1)
    rloc_Ind = find(position_time_stamps * 1000 > position_time_stamps_milliseconds(ripple_index(ripple_number, 1)) & ...
        position_time_stamps * 1000 < position_time_stamps_milliseconds(ripple_index(ripple_number, 2)));
    running_speed_at_ripple(ripple_number) = running_speed(rloc_Ind(1), 1);
end

ripple_index = ripple_index(running_speed_at_ripple < 4 & num_spikes_per_ripple > 0, :);
end

function [x] = to_milliseconds(x)
x = round(x * 1000);
end