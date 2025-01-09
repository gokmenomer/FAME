function plotAsymmetricGaussians(left_sigmas, right_sigmas, centers)
    % plotAsymmetricGaussians Plots asymmetric Gaussian membership functions.
    %
    %   plotAsymmetricGaussians(left_sigmas, right_sigmas, centers)
    %
    %   Inputs:
    %       left_sigmas  - m x n matrix of left sigmas for Gaussians
    %       right_sigmas - m x n matrix of right sigmas for Gaussians
    %       centers      - m x n matrix of centers for Gaussians

    % Extract data from dlarray if necessary
    if isa(centers, 'dlarray')
        centers = extractdata(centers);
    end
    if isa(left_sigmas, 'dlarray')
        left_sigmas = extractdata(left_sigmas);
    end
    if isa(right_sigmas, 'dlarray')
        right_sigmas = extractdata(right_sigmas);
    end

    % Validate input dimensions
    [m, n] = size(centers);
    assert(all(size(left_sigmas) == [m, n]), 'left_sigmas must be of size m x n');
    assert(all(size(right_sigmas) == [m, n]), 'right_sigmas must be of size m x n');

    % Assuming n=1 since we're plotting per subnet; adjust if needed
    % Here, we'll handle n=1. If n>1, you might want to loop through columns.

    % Define x range
    left_max = 3 * max(left_sigmas, [], 1);
    right_max = 3 * max(right_sigmas, [], 1);

    x_min = min(centers(:, 1)) - left_max(1);
    x_max = max(centers(:, 1)) + right_max(1);
    x_vals = linspace(x_min, x_max, 1000)'; % Column vector for matrix operations

    % Initialize Y matrix to store all Gaussian values
    Y = zeros(length(x_vals), m);

    % Vectorize Gaussian computation using logical indexing
    for i = 1:m
        mask = x_vals <= centers(i, 1);
        % Compute y values for x <= center
        Y(mask, i) = exp(-0.5 * ((x_vals(mask) - centers(i, 1)).^2) / (left_sigmas(i, 1)^2));
        % Compute y values for x > center
        Y(~mask, i) = exp(-0.5 * ((x_vals(~mask) - centers(i, 1)).^2) / (right_sigmas(i, 1)^2));
    end

    % Plot all Gaussians
    plot(x_vals, Y, 'LineWidth', 1.5);

    % Optional: Add legends only if number of Gaussians is small
    if m <= 10 % Adjust threshold as needed
        legend(arrayfun(@(i) sprintf('Gaussian %d', i), 1:m, 'UniformOutput', false), 'Location', 'Best');
    end

    % Customize plot
    xlabel('x');
    ylabel('Membership Value');
    grid on;
end
