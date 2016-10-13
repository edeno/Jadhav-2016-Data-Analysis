function y = normalize_distribution(x)
% Makes a distribution a probability distribution by making sure it sums to
% 1
y = x ./ sum(x);
end