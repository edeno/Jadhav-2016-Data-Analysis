function [initial_conditions] = get_initial_conditions(linear_distance_bins)
% Where the replay trajectory starts. Assume the replay event is equally 
% likely to happen at any position in the maze
num_linear_distance_bins = length(linear_distance_bins);
linear_distance_bin_size = linear_distance_bins(2) - linear_distance_bins(1);
%% P(x0|I)
% Gaussian with probability mass at center arm reflecting that the replay
% trajectory is likely to start at the center arm
Px_I{1} = normpdf(linear_distance_bins, 0, linear_distance_bin_size * 2);
Px_I{1} = Px_I{1} ./ sum(Px_I{1});
% Gaussian with probability mass everywhere but at the center arm
% reflecting that the replay trajectory is likely to start everywhere
% except at the center arm
Px_I{2} = max(Px_I{1}) * ones(1, num_linear_distance_bins) - Px_I{1};
Px_I{2} = Px_I{2} ./ sum(Px_I{2});

%% P(x0)=P(x0|I)P(I);
prior_probability_of_state = 1/4;
initial_conditions(:, 1) = prior_probability_of_state * Px_I{1}'; % outbound forward
initial_conditions(:, 2) = prior_probability_of_state * Px_I{2}'; % outbound reverse
initial_conditions(:, 3) = prior_probability_of_state * Px_I{2}'; % inbound forward
initial_conditions(:, 4) = prior_probability_of_state * Px_I{1}'; % inbound reverse
end