function varargout = inference(Train, Test, yTrue_train, yTrue_test, number_of_rules, number_inputs,number_outputs, Learnable_parameters, output_membership_type,tnorm, nF, dr_method, return_firestrength)

    if nargin < 13 || isempty(return_firestrength)
        %% Inference
        yPred_train = fismodel(Train.inputs, number_of_rules, number_inputs,number_outputs,length(Train.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method);
        yPred_test = fismodel(Test.inputs, number_of_rules, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method);

    else
        %% Inference
        [yPred_train, firing_train] = fismodel(Train.inputs, number_of_rules, number_inputs,number_outputs,length(Train.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method, return_firestrength);
        [yPred_test, firing_test] = fismodel(Test.inputs, number_of_rules, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method, return_firestrength);

    end


    yPred_train = reshape(yPred_train, [number_outputs, size(Train.inputs,3)]);
    yPred_test = reshape(yPred_test, [number_outputs, size(Test.inputs,3)]);

    train_RMSE = rmse(yPred_train', yTrue_train')
    test_RMSE = rmse(yPred_test', yTrue_test')


    if nargin < 13 || isempty(return_firestrength)
        varargout{1} = yPred_train;
        varargout{2} = yPred_test;
    else
        varargout{1} = train_RMSE;
        varargout{2} = test_RMSE;
        varargout{3} = firing_train;
        varargout{4} = firing_test;
    end

end