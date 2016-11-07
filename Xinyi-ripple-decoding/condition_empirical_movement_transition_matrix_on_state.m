function [empirical_movement_transition_matrix] = condition_empirical_movement_transition_matrix_on_state(linear_distance_bins, linear_distance, state_index)
%% calculate emipirical movement transition matrix, then Gaussian smoothed
num_linear_distance_bins = length(linear_distance_bins);
stateM = zeros(num_linear_distance_bins);
[~, state_bin] = histc(linear_distance(state_index), linear_distance_bins);
state_disM = [state_bin(1:end-1) state_bin(2:end)];
for bin_ind = 1:num_linear_distance_bins
    sp0 = state_disM(state_disM(:, 1) == bin_ind, 2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if ~isempty(sp0)
        stateM(:, bin_ind) = histc(sp0, linspace(1, num_linear_distance_bins, num_linear_distance_bins)) ./ size(sp0, 1);
    else
        stateM(:, bin_ind) = zeros(1, num_linear_distance_bins);
    end
end

%%%if too many zeros:
for bin_ind = 1:num_linear_distance_bins
    if sum(stateM(:, bin_ind)) == 0
        stateM(:, bin_ind) = 1 / num_linear_distance_bins;
    else
        stateM(:, bin_ind) = stateM(:,bin_ind) ./ sum(stateM(:, bin_ind));
    end
end

[dx, dy] = meshgrid([-1:1]);
sigma = 0.5;
normalizing_weight = gaussian(sigma, dx, dy) / sum(sum(gaussian(sigma, dx, dy))); %normalizing weights
stateM_gaussian_smoothed = conv2(stateM, normalizing_weight, 'same'); %gaussian smoothed
empirical_movement_transition_matrix = stateM_gaussian_smoothed * diag(1 ./ sum(stateM_gaussian_smoothed, 1)); %normalized to confine probability to 1

end

function [value] = gaussian(sigma, x, y)
value = exp(-(x.^2 + y.^2) / 2 / sigma^2); %gaussian
end