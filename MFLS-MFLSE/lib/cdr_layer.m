function fc_output = cdr_layer(mini_batch_inputs, learnable_parameters, number_inputs, mbs, nF, dr_method)

    if dr_method == "cdr" || (dr_method == "dr")

        mini_batch_inputs_cdr = permute(mini_batch_inputs, [2 3 1]);

        x_centered = mini_batch_inputs_cdr ;

        fc_output = learnable_parameters.WA0(1:number_inputs,:)' * x_centered;

        bias_replicated = repmat(reshape(learnable_parameters.WA0(number_inputs+1,:), [nF,1]), 1, size(fc_output, 2));

        fully_connected_output = fc_output + bias_replicated;

        fc_output = permute(fully_connected_output, [3 1 2]);
    else
        fc_output = mini_batch_inputs;
    end

end