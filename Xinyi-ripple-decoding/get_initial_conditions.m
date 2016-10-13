function [initial_conditions] = get_initial_conditions(linear_distance_bins)
%% Where the replay trajectory starts
num_linear_distance_bins = length(linear_distance_bins);
linear_distance_bin_size = linear_distance_bins(2) - linear_distance_bins(1);
%P(x0|I);
Px_I{1} = exp(-linear_distance_bins.^2  ./ (2 * (2 * linear_distance_bin_size)^2));
Px_I{1} = Px_I{1} ./ sum(Px_I{1});
Px_I{2} = max(Px_I{1}) * ones(1, num_linear_distance_bins) - Px_I{1};
Px_I{2} = Px_I{2} ./ sum(Px_I{2});

%P(x0)=P(x0|I)P(I);
initial_conditions(:, 1) = 0.25 * Px_I{1}'; % outbound forward
initial_conditions(:, 2) = 0.25 * Px_I{2}'; % outbound reverse
initial_conditions(:, 3) = 0.25 * Px_I{2}'; % inbound forward
initial_conditions(:, 4) = 0.25 * Px_I{1}'; % inbound reverse
end