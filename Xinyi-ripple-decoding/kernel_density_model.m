function [markAll, time, mark0, procInd1] = kernel_density_model(animal, day, tetrode_number, ...
    linear_position_time, mdel, sxker, xs, xtrain, num_time_points, dt)
mark_data_file = load(get_mark_filename('bond', day, tetrode_number));
params = mark_data_file.filedata.params;
ind = find(params(:, 1) / 1E4 >= linear_position_time(1) & ...
    params(:, 1) / 1E4 <= linear_position_time(end));
time = params(ind, 1);
mark0 = params(ind, 2:5); % tetrode wire maxes
time2= time / 1E4;
spikeT0 = time2;
[procInd0,procInd1] = histc(spikeT0, linear_position_time);
procInd = find(procInd0);
spikeT = linear_position_time(procInd);
spike = procInd0';
[~, rawInd0] = histc(spikeT0, time2);
markAll(:, 1) = procInd1;
markAll(:, 2:5) = mark0(rawInd0(rawInd0~=0), :);
ms = min(min(markAll(:, 2:5))):mdel:max(max(markAll(:, 2:5)));
occupancy = normpdf(xs' * ones(1, num_time_points), ones(length(xs), 1) * xtrain, sxker) * ones(num_time_points, length(ms));
%occ: columns are identical; occupancy based on position; denominator
gaussian_kernel_estimator_position = normpdf(xs' * ones(1,length(spikeT0)), ones(length(xs), 1) * xtrain(procInd1), sxker);
%Xnum: Gaussian kernel estimators for position
Lintegral = sum(gaussian_kernel_estimator_position, 2) ./ occupancy(:,1) ./ dt; %integral
Lintegral = Lintegral ./ sum(Lintegral);
end

function [filename] = get_mark_filename(animal, day, tetrode_number)
    filename = sprintf('bond_data/%s%02d-%02d_params.mat', animal, day, tetrode_number);
end