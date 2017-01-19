function [mark_spikes_to_linear_distance_time_bins_index] = kernel_density_model( ...
    mark_spike_times, linear_distance_time)
mark_spike_time_in_seconds = mark_spike_times / 1E4;
[~, mark_spikes_to_linear_distance_time_bins_index] = histc(mark_spike_time_in_seconds, linear_distance_time);
end