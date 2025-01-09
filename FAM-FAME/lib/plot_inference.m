function results = plot_inference(number_of_rules, number_inputs, number_outputs, length_train, Learnable_parameters, output_membership_type, nF, dr_method, plot_index)

    %% Inference for Plotting
    X_for_plot = 0:0.01:1;
    X_index = X_for_plot;
    X_for_plot = repmat(X_for_plot, number_inputs, 1);
    X_for_plot = dlarray(permute(X_for_plot, [3 1 2]));

    results = model_subnets(X_for_plot, number_of_rules, number_inputs, number_outputs, length_train, Learnable_parameters, output_membership_type, nF, dr_method);
    figure
    plot(X_index, results(plot_index, :)); hold on;

end
