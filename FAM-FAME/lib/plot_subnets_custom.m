function plot_subnets_custom(Learnable_parameters, max_min_plot, zscore_plot, add_max_min, Train, nF, prev_learnable_parameters)
    % Function to plot symmetric Gaussian membership functions for all subnets
    %
    % Parameters:
    %   Learnable_parameters   - Structure containing learnable parameters with fields
    %                            'subnet1', 'subnet2', ..., each having 'input_centers' and 'input_sigmas'
    %   max_min_plot           - Boolean flag to plot global max/min vertical lines
    %   zscore_plot            - Boolean flag to plot z-score based lines
    %   add_max_min            - Boolean flag to add additional max/min lines per subnet
    %   Train                  - Structure containing training data, specifically 'inputs'
    %   nF                     - Number of features or subnets
    %   prev_learnable_parameters - Previous learnable parameters for additional computations

    % Identify subnet fields (assuming they start with 'subnet')
    all_fields = fieldnames(Learnable_parameters);
    subnet_fields = all_fields(startsWith(all_fields, 'subnet'));

    num_subnets = numel(subnet_fields);

    % Initialize cell arrays to store centers and sigmas
    all_centers = cell(1, num_subnets);
    all_sigmas = cell(1, num_subnets);

    global_x_min = inf; % To calculate global x-axis range
    global_x_max = -inf;

    % Initialize variables for max/min data
    max_data = [];
    min_data = [];

    % Process each subnet to extract centers and sigmas
    for k = 1:num_subnets
        field_name = subnet_fields{k};
        current_subnet = Learnable_parameters.(field_name);

        % Extract centers and sigmas
        centers = gather(extractdata(current_subnet.input_centers));
        sigmas = gather(extractdata(current_subnet.input_sigmas));

        % Store the results
        all_centers{k} = centers;
        all_sigmas{k} = sigmas;

        % Update global x-axis limits
        sigma_max = 3 * max(sigmas, [], 'all');
        global_x_min = min(global_x_min, min(centers, [], 'all') - sigma_max);
        global_x_max = max(global_x_max, max(centers, [], 'all') + sigma_max);
    end

    % Ensure valid x-axis limits
    % global_x_min = global_x_min)); % Convert to numeric if necessary
    % global_x_max = gather(extractdata(global_x_max)); % Convert to numeric if necessary

    if global_x_min >= global_x_max
        error('Invalid global x-axis limits: Ensure centers and sigmas are correct.');
    end

    % Create a figure
    figure('Name', 'Symmetric Gaussians for All Subnets');
    clf; % Clear current figure

    % Plot for each subnet
    for k = 1:num_subnets
        current_centers = all_centers{k};
        current_sigmas = all_sigmas{k};

        % Define x range
        x_vals = linspace(global_x_min, global_x_max, 10000)'; % Column vector

        % Initialize Y matrix for Gaussian values
        m = numel(current_centers);
        Y = zeros(length(x_vals), m);

        % Compute Gaussian membership values using custom_gaussmf
        for i = 1:m
            Y(:, i) = custom_gaussmf(x_vals, current_sigmas(i), current_centers(i));
        end

        % Create subplot
        subplot(num_subnets, 1, k);
        hold on;

        % Plot Gaussian membership functions
        plot(x_vals, Y, 'LineWidth', 1.5);

        % Add vertical dashed lines if max_min_plot is true
        if max_min_plot
            % Plot global x = 0 and x = 1 lines
            xline(0, 'k--', 'LineWidth', 1.2, 'DisplayName', 'x = 0');
            xline(1, 'k--', 'LineWidth', 1.2, 'DisplayName', 'x = 1');
        end

        % Add z-score based lines if zscore_plot is true
        if zscore_plot
            % Find z-score based max and min for this subnet
            max_data = max(Train.inputs, [], 3);
            min_data = min(Train.inputs, [], 3);
            min_data = gather(extractdata(min_data));
            max_data = gather(extractdata(max_data));
            % Use max and min values corresponding to this subnet (assuming min_data/max_data are nF×1 vectors)
            xline(min_data(k), 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Min (nF = %d)', k));
            xline(max_data(k), 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Max (nF = %d)', k));
            global_x_min = min(min_data(k), global_x_min);
            global_x_max = max(max_data(k), global_x_max);
        end

        % Add additional max/min lines if add_max_min is true
        if add_max_min
            data = cdr_layer(Train.inputs, prev_learnable_parameters, size(Train.inputs,2), nF, "cdr");
            max_data = max(data, [], 3);
            min_data = min(data, [], 3);
            min_data = gather(extractdata(min_data));
            max_data = gather(extractdata(max_data));
            % Use max and min values corresponding to this subnet (assuming min_data/max_data are nF×1 vectors)
            xline(min_data(k), 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Min (nF = %d)', k));
            xline(max_data(k), 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Max (nF = %d)', k));
            global_x_min = min(min_data(k), global_x_min);
            global_x_max = max(max_data(k), global_x_max);
            xlim([min_data(k), max_data(k)]);
        end

        % Customize subplot
        xlabel(sprintf('\\it z_{%d}', k));
        ylabel('\mu');
        % title(sprintf('Symmetric Gaussian - Subnet %d', k));
        grid on;
        hold off;

        % Set x-axis limits to global range
        % xlim([global_x_min, global_x_max]);
    end
    exportgraphics(gcf, 'gaussmf_seed_2_l_f_cdr_4.png', 'Resolution', 300, 'ContentType', 'auto', 'BackgroundColor', 'none');

    % Add a super title to the figure
    % sgtitle('Symmetric Gaussian Membership Functions for All Subnets');
end

function output = custom_gaussmf(x, s, c)
    % Custom Gaussian membership function
    % x: input values
    % s: sigma (standard deviation)
    % c: center of the Gaussian
    exponent = -0.5 * ((x - c).^2 ./ s.^2); % Calculate exponent
    output = exp(exponent); % Compute Gaussian values
end

%%% cdr4l2 0.6602
%%% cdr4l_f 0.6570
