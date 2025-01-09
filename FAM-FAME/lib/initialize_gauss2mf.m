function [learnable_parameters] = initialize_gauss2mf(input_data, output_data, number_of_rule, output_type)
    %% centers
    number_inputs = size(input_data,2);
    number_outputs = size(output_data,2);

    data = input_data;
    data = extractdata(permute(data,[3 2 1]));

    %% Step 1: Initialize the most left Gaussian center as a learnable parameter
    % Set the initial value for the first center
    initial_center = min(data(:));  % e.g., mean of data

    %% Step 2: Initialize sigmas based on data standard deviation with added rule
    % Calculate standard deviation for each feature
    delta_dist = max(data) - min(data);
    delta_gauss = delta_dist / (number_of_rule );
    sigma_gauss = delta_gauss / 4;
    initial_center = initial_center + (2*sigma_gauss);
    s = sigma_gauss;
    s(s == 0) = 1; % Handle zero std casestest_num
    learnable_sigmas = repmat(s, number_of_rule + 1, 1);
    learnable_parameters.input_sigmas = dlarray(learnable_sigmas); % Make sigmas learnable
    learnable_parameters.leftmost_center = dlarray(initial_center); % Make it learnable



    if output_type == "singleton"

        c = rand(number_of_rule,number_outputs)*0.01;
        learnable_parameters.singleton.c = dlarray(c);

    elseif output_type == "linear"

        a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
        learnable_parameters.linear.a = dlarray(a);

        b = rand(number_of_rule*number_outputs,1)*0.01;
        learnable_parameters.linear.b = dlarray(b);
    end

    end