function subnets = initialize_model(input_data, output_data,number_of_rule, output_membership_type, nF, dr_method, gauss2MF)


number_inputs = size(input_data, 2);

subnets = struct;

%cdr init (PCA)
if dr_method ~= "none"
    Train_temp = permute(input_data, [3 2 1]);
    [~, WA0] = pca_init(extractdata(Train_temp), nF);
    subnets.WA0 = dlarray(WA0);
    XTrainA = cdr_layer(permute(Train_temp(:,1:number_inputs), [3 2 1 ]), subnets, number_inputs, nF, dr_method);
    XTrainA = permute(XTrainA, [3 2 1]);
    number_inputs = nF; % check
    input_data = XTrainA;
    input_data = dlarray(permute(input_data, [3 2 1]));

end




for i = 1:number_inputs
    if gauss2MF
        subnet = initialize_gauss2mf(input_data(:, i, :), output_data, number_of_rule, output_membership_type);
    else
        subnet = initialize_Glorot_Kmeans(input_data(:, i, :), output_data, number_of_rule, output_membership_type);
    end
        subnets.("subnet" + i) = subnet;
end

subnets.("output_layer") = struct;
sz = [1, number_inputs];
subnets.("output_layer").Weights = initializeGlorot(sz, 1, number_inputs);
subnets.("output_layer").Bias = initializeZeros([1 1]);




end
%%
function weights = initializeGlorot(sz,numOut,numIn,className)

arguments
    sz
    numOut
    numIn
    className = 'single'
end

Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end
%%
function parameter = initializeZeros(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end
