function [mark_spike_times, marks, mark_spikes_to_linear_position_time_bins_index] = kernel_density_model( ...
    mark_spike_times, marks, linear_position_time)

mark_spike_time_in_seconds = mark_spike_times / 1E4;

ind = mark_spike_time_in_seconds >= linear_position_time(1) & ...
    mark_spike_time_in_seconds <= linear_position_time(end);

mark_spike_times = mark_spike_times(ind);
mark_spike_time_in_seconds = mark_spike_times / 1E4;

[~, mark_spikes_to_linear_position_time_bins_index] = histc(mark_spike_time_in_seconds, linear_position_time);
end