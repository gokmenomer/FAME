function output = matrix_fuzzification_layer(x, membership_type, learnable_parameters, number_of_rule, number_inputs, mbs)
% calculating fuzzified values
%
% @param output -> output
%
%       (mfc,ic,mbs) tensor
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> fuzzified value of the first input with first Membership
%       Function (MF) of that input of first element of the batch
%
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> first input of the first element of the batch
%
% @param input 2 -> membership_type
%
%       a string
%       type of the membership function
%       it is gaussmf for now but gauss2mf will be added
%
% @param input 3 -> learnable_parameters
%
%       struct
%       consist of sigma and center values of each mf
%
% @param input 4 -> number_of_rule
%
%       constant
%       number of membership function for inputs
%       In other words number of rules of FLS 
%
% @param input 5 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 6 -> mbs
%
%       constant
%       number of mini-batch size
%

output = zeros(number_of_rule, number_inputs, mbs,"gpuArray");


if(membership_type == "gaussmf")

    input_matrix = repmat(x, number_of_rule, 1, 1); % expanding the input matrix by the number of rules

    if isfield(learnable_parameters, 'leftmost_center') % For FAME
        [centers, left_sigmas, right_sigmas] = calculate_centers(learnable_parameters.leftmost_center,learnable_parameters.input_sigmas);
        output = custom_asymmetric_gaussmf(input_matrix, left_sigmas, right_sigmas, centers); % calculating fuzzified values
    else % For FAM
        output = custom_gaussmf(input_matrix, learnable_parameters.input_sigmas, learnable_parameters.input_centers); % calculating fuzzified values
    end

elseif(membership_type ~= "gaussmf") %for future expansion
else %for future expansion
end

output = dlarray(output);


end


%%

% Custom Gaussian function
function output = custom_gaussmf(x, s, c) % s -> sigma of the Gauss MF / c -> center of the Gauss MF
    exponent = -0.5 * ((x - c).^2 ./ s.^2); % calculating exponent of the Gauss MF
    output = exp(exponent);
end

function output = custom_asymmetric_gaussmf(x, left_sigma, right_sigma, c)
    % Custom Gaussian Membership Function with asymmetric (left-right) sigmas
    % x           -> input values [5, 4, batch size]
    % left_sigma  -> sigma to the left of the center [5, 4]
    % right_sigma -> sigma to the right of the center [5, 4]
    % c           -> center of the Gauss MF [5, 4]

    % Expand left_sigma, right_sigma, and c to match x dimensions
    left_sigma = repmat(left_sigma, 1, 1, size(x, 3));
    right_sigma = repmat(right_sigma, 1, 1, size(x, 3));
    c = repmat(c, 1, 1, size(x, 3));

    % Initialize output array
    output = dlarray(zeros(size(x)));

    % Calculate membership values based on left and right sigmas
    left_indices = x <= c;
    right_indices = x > c;

    % Left side calculation (x <= c)
    output(left_indices) = exp(-0.5 * ((x(left_indices) - c(left_indices)).^2 ./ left_sigma(left_indices).^2));

    % Right side calculation (x > c)
    output(right_indices) = exp(-0.5 * ((x(right_indices) - c(right_indices)).^2 ./ right_sigma(right_indices).^2));
end