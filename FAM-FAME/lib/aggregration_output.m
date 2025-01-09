function y_pred = aggregration_output(subnet_outputs, learnable_parameters, gam_aggregration_method)

    if gam_aggregration_method == "weighted"
        y_pred = learnable_parameters.output_layer.Weights*subnet_outputs + learnable_parameters.output_layer.Bias;
    elseif gam_aggregration_method == "sum"
        y_pred = sum(subnet_outputs, 1);
    end
    y_pred = permute(y_pred, [1 3 2]);

end