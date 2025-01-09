function plot_subnet_gauss2mf(Learnable_parameters, max_min_plot, zscore_plot, add_max_min, Train, nF, prev_learnable_parameters)
   
    % Identify subnet fields (assuming they start with 'subnet')
    all_fields = fieldnames(Learnable_parameters);
    subnet_fields = all_fields(startsWith(all_fields, 'subnet'));

    num_subnets = numel(subnet_fields);

    % Initialize cell arrays to store results
    all_centers = cell(1, num_subnets);
    all_left_sigmas = cell(1, num_subnets);
    all_right_sigmas = cell(1, num_subnets);

    global_x_min = inf; % To calculate global x-axis range
    global_x_max = -inf;

    % Process each subnet
    for k = 1:num_subnets
        field_name = subnet_fields{k};
        current_subnet = Learnable_parameters.(field_name);

        % Calculate centers, left sigmas, and right sigmas
        [centers, left_sigmas, right_sigmas] = calculate_centers( ...
            gather(extractdata(current_subnet.leftmost_center)), ...
            gather(extractdata(current_subnet.input_sigmas)));

        % Store the results
        all_centers{k} = centers;
        all_left_sigmas{k} = left_sigmas;
        all_right_sigmas{k} = right_sigmas;

        % Update global x-axis limits
        left_max = 3 * max(left_sigmas, [], 'all');
        right_max = 3 * max(right_sigmas, [], 'all');
        global_x_min = min(global_x_min, min(centers, [], 'all') - left_max);
        global_x_max = max(global_x_max, max(centers, [], 'all') + right_max);
    end

    % Ensure valid x-axis limits
    global_x_min = gather(extractdata(global_x_min)); % Convert to numeric
    global_x_max = gather(extractdata(global_x_max)); % Convert to numeric

    if global_x_min >= global_x_max
        error('Invalid global x-axis limits: Ensure centers and sigmas are correct.');
    end

    % Create a figure
    figure('Name', 'Asymmetric Gaussians for All Subnets');
    clf; % Clear current figure

    % Plot for each subnet
    for k = 1:num_subnets
        current_centers = all_centers{k};
        current_left_sigmas = all_left_sigmas{k};
        current_right_sigmas = all_right_sigmas{k};

        % Define x range
        x_vals = linspace(global_x_min, global_x_max, 10000)'; % Column vector

        % Initialize Y matrix for Gaussian values
        [m, ~] = size(current_centers);
        Y = zeros(length(x_vals), m);

        % Compute Gaussian membership values
        for i = 1:m
            mask = x_vals <= current_centers(i);
            % Compute y values for x <= center
            Y(mask, i) = exp(-0.5 * ((x_vals(mask) - current_centers(i)).^2) / (current_left_sigmas(i)^2));
            % Compute y values for x > center
            Y(~mask, i) = exp(-0.5 * ((x_vals(~mask) - current_centers(i)).^2) / (current_right_sigmas(i)^2));
        end

        % Create subplot
        subplot(num_subnets, 1, k);
        hold on;

        % Plot Gaussian membership functions
        plot(x_vals, Y, 'LineWidth', 1.5);

        % Add vertical dashed lines if add_max_min is true
        if max_min_plot
            xline(0, 'k--', 'LineWidth', 1.2, 'DisplayName', 'x = 0');
            xline(1, 'k--', 'LineWidth', 1.2, 'DisplayName', 'x = 1');
        end

        if zscore_plot
            % Find max and min values for z-score plot
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

        % Add vertical dashed lines if add_max_min is true
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
        % title(sprintf('Asymmetric Gaussian - Subnet %d', k));
        grid on;
        hold off;

        % Set x-axis limits to global range
        % xlim([global_x_min, global_x_max]);
    end
    
    % exportgraphics(gcf, 'gauss2mf_seed_2_l2_cdr_4.png', 'Resolution', 300, 'ContentType', 'auto', 'BackgroundColor', 'none');

    % Add a super title to the figure
    % sgtitle('Asymmetric Gaussian Membership Functions for All Subnets');
end

% For untrained CDR 
% plot_subnet_gauss2mf(prev_learnable_parameters, 0, 0, 1, Train, nF, prev_learnable_parameters)
% For trained CDR
% plot_subnet_gauss2mf(Learnable_parameters, 0, 0, 1, Train, nF, prev_learnable_parameters)

% For untrained Vanilla max-min normalized
% plot_subnet_gauss2mf(prev_learnable_parameters, 1, 0, 0, Train, nF, prev_learnable_parameters)
% For trained Vanilla
% plot_subnet_gauss2mf(Learnable_parameters, 1, 0, 0, Train, nF, prev_learnable_parameters)

% For untrained Vanilla z-score normalized
% plot_subnet_gauss2mf(prev_learnable_parameters, 0, 1, 0, Train, nF, prev_learnable_parameters)
% For trained Vanilla z-score normalized
% plot_subnet_gauss2mf(Learnable_parameters, 0, 1, 0, Train, nF, prev_learnable_parameters)


%%% cdr4l2 0.6660
%%% cdrl_f 0.6522
