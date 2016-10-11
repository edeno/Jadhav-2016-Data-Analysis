function [rippleI, ripple_index, position_time_stamps, position_time_stamps_binned] = get_ripple_index(pos, ripplescons, spikes, tetrode_index, neuron_index)
%% calculate ripple starting and end times
position_time_stamps = pos.data(:,1); %time stamps for animal's trajectory
position_time_stamps_binned = round(position_time_stamps(1) * 1000):1:round(position_time_stamps(end) * 1000); %binning time stamps at 1 ms

ripple_start_time = ripplescons{1}.starttime;
ripple_end_time = ripplescons{1}.endtime;
traj_Ind = find(ripplescons{1}.maxthresh > 4);
ripple_start_time = ripple_start_time(traj_Ind);
ripple_end_time = ripple_end_time(traj_Ind);
ripple_index = [round(ripple_start_time * 1000) - position_time_stamps_binned(1) - 1, ...
    round(ripple_end_time * 1000) - position_time_stamps_binned(1) - 1]; %index for ripple segments

for neuron_ind = 1:size(tetrode_index, 2)
    spike_times = spikes{tetrode_index(neuron_ind)}{neuron_index(neuron_ind)}.data(:,1); %spiking times for tetrode j, cell i
    binned_spike_times = round(spike_times * 1000); %binning spiking times at 1 ms
    [sptrain2_list{neuron_ind}, ~] = ismember(position_time_stamps_binned, binned_spike_times); %sptrain2: spike train binned at 1 ms instead of 33.4ms (sptrain0)
end

for k = 1:size(ripple_index,1)
    spike_r = [];
    for neuron_ind = 1:size(tetrode_index, 2)
        sptrain2 = sptrain2_list{neuron_ind};
        spike_r = [spike_r; sptrain2(ripple_index(k, 1):ripple_index(k ,2))];
    end
    spike_r_all{k} = spike_r;
end

for k = 1:size(ripple_index, 1)
    spike_r = spike_r_all{k};
    sumR(k) = sum(spike_r(:));
end
rippleI = find(sumR > 0);

end