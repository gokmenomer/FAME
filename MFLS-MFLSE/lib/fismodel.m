function varargout = fismodel(mini_batch_inputs, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, output_membership_type, tnorm, nF, dr_method, return_firestrength)

% Fuzzy Inferance System Model Structure - Fuzzy Logic System Model Structure
%
% @param output -> ypred
%
%       (1,oc,mbs) tensor
%       crisp output of the FLS
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> firts crisp output of the first element of the batch
%
% @param input 1 -> mini_batch_inputs
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
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
% @param input 8 -> tnorm
%
%       a string
%       T-norm operator type
%       2 options are available for now: 
%       "product" , "HTSK"      
%       you can use other tnorm operators such as min check
%       "firing_strength_calculation_layer" function
%
%       some other types are excluded since not published yet!
%
% @param input 9 -> nF
%
%       constant
%       Dimension for CDR layer
%
% @param input 10 -> dr_method
%
%       a string
%       dimensionality reduction method used in the FLS
%
% @param input 11 -> return_firestrength (optional)
%
%       boolean
%       If true, the function returns both ypred and firestrength
%       If false, only ypred is returned (default behavior)


% Infer firestrength based on the T-norm method

    number_inputs = size(mini_batch_inputs, 2);
    fully_connected_output = cdr_layer(mini_batch_inputs, learnable_parameters, number_inputs, mbs, nF, dr_method);
    fuzzified = matrix_fuzzification_layer(fully_connected_output, learnable_parameters, number_of_rule, number_inputs, mbs);
    firestrength = firing_strength_calculation_layer(fuzzified, tnorm);
    normalized = firing_strength_normalization_layer(firestrength);


    % Dimensionality reduction method check
    if dr_method == "cdr"
        mini_batch_inputs = fully_connected_output;
    end

    % Defuzzification to get final output
    ypred = multioutput_defuzzification_layer(mini_batch_inputs, normalized, learnable_parameters, number_outputs, output_membership_type);

    % Check if the user wants to return both ypred and firestrength
    if nargin < 11 || isempty(return_firestrength)
        varargout{1} = ypred;  % Return only ypred
    else
        varargout{1} = ypred;  % Return ypred
        varargout{2} = normalized;  % Return firestrength if requested
    end

end
