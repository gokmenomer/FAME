function ypred = fismodel(mini_batch_inputs, number_of_rule, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type)

% Fuzzy Inferance System Model Structure - Fuzzy Logic System Model Structure
%
% @param output -> ypred
%
%       (1,oc,mbs) tensor
%       crisp output of the FLS
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> first crisp output of the first element of the batch
%
% @param input 1 -> mini_batch_inputs
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> first input of the first element of the batch
%
% @param input 2 -> number_of_rule
%
%       constant
%       number of Membership Function (MF) for inputs
%       In other words number of rules of the FLS
%
% @param input 3 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 4 -> number_outputs
%
%       constant
%       number of outputs
%
% @param input 5 -> mbs
%
%       constant
%       number of mini-batch size
%
% @param input 6 -> learnable_parameters
%
%       struct
%       consist parameters of antecedent and consequent MFs
%
% @param input 7 -> output_membership_type
%
%       a string
%       consequent MF type
%       2 options are available for now: 
%       "singleton" , "linear" 
%       other types are excluded since not published yet!
%

    fuzzified = matrix_fuzzification_layer(mini_batch_inputs, "gaussmf", learnable_parameters, number_of_rule, number_inputs, mbs);
    firestrength = firing_strength_calculation_layer(fuzzified, "product");
    normalized = firing_strength_normalization_layer(firestrength);
    ypred = multioutput_defuzzification_layer(mini_batch_inputs, normalized, learnable_parameters,number_outputs, output_membership_type);

end