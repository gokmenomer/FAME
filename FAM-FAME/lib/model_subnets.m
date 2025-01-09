function [subnet_outputs] = model_subnets(mini_batch_inputs, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, output_membership_type, nF, dr_method)


subnet_outputs = [];

if dr_method ~="none"
    mini_batch_inputs = cdr_layer(mini_batch_inputs, learnable_parameters, number_inputs, nF, dr_method);
    number_inputs = nF;
end

for i = 1: number_inputs
    subnet = learnable_parameters.("subnet" + i);
    subnet_output= fismodel(mini_batch_inputs(:, i, :), number_of_rule, 1, number_outputs, mbs, subnet, output_membership_type);
    subnet_output = permute(subnet_output, [1 3 2]);
    subnet_outputs = [subnet_outputs;subnet_output];
end

end