function [stateM_normalized_gaussian] = condition_empirical_movement_transition_matrix_on_state(stateV, vecLF, is_state)
stateV_length = length(stateV);
stateM_I = zeros(stateV_length);
vecLF_seg = vecLF(is_state, :);
[~, state_bin] = histc(vecLF_seg(:, 2), stateV);
state_disM = [state_bin(1:end-1) state_bin(2:end)];
stateM_seg = zeros(stateV_length);
for stateV_ind=1:stateV_length
    sp0 = state_disM(state_disM(:,1) == stateV_ind, 2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if ~isempty(sp0)
        stateM_seg(:, stateV_ind) = histc(sp0, linspace(1, stateV_length, stateV_length)) ./ size(sp0, 1);
    else
        stateM_seg(:, stateV_ind) = zeros(1, stateV_length);
    end
end
stateM_I = stateM_I + stateM_seg;
%%%if too many zeros:
for i=1:stateV_length
    if sum(stateM_I(:, i)) == 0
        stateM_I(:, i) = 1 / stateV_length;
    else
        stateM_I(:, i) = stateM_I(:,i) ./ sum(stateM_I(:, i));
    end
end

[dx, dy] = meshgrid([-1:1]);
sigma = 0.5;
normalizing_weight = gaussian(sigma, dx, dy) / sum(sum(gaussian(sigma, dx, dy))); %normalizing weights
stateM_gaussian_smoothed = conv2(stateM_I, normalizing_weight, 'same'); %gaussian smoothed
stateM_normalized_gaussian = stateM_gaussian_smoothed * diag(1 ./ sum(stateM_gaussian_smoothed, 1)); %normalized to confine probability to 1

end

function [value] = gaussian(sigma, x, y)
value = exp(-(x.^2 + y.^2) / 2 / sigma^2); %gaussian
end