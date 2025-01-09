function [centers, left_sigmas, right_sigmas] = calculate_centers(leftmost_center, sigmas)

    % Number of centers (m) is one less than the length of sigmas
    m = size(sigmas, 1) - 1;
    n = size(sigmas, 2); % Number of columns for different batches/experiments
    sigmas = abs(sigmas);
    % Calculate the left and right sigmas for each Gaussian
    left_sigmas = sigmas(1:end-1, :);
    right_sigmas = sigmas(2:end, :);

    % Initialize centers matrix
    centers = dlarray(zeros(m, n), 'CB'); % Assuming batch mode for the second dimension

    % Set the first center based on leftmost_center
    centers(1, :) = leftmost_center;

    % Calculate each subsequent center
    for i = 2:m
        centers(i, :) = centers(i - 1, :) + 4 * sigmas(i, :);
    end
end
