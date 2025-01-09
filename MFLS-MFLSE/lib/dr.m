function [Train, Test, XTrainA, Learnable_parameters, nF] = dr(Train_temp, Test_temp, dr_method, nF, number_inputs, number_outputs, training_num, test_num, number_of_rules, output_membership_type, gauss2MF)
    % Initialize the Learnable_parameters structure
    Learnable_parameters = struct;
    if dr_method == "none"
        XTrainA = Train_temp(:,1:number_inputs)';
        nF = number_inputs;
        Train.inputs = reshape(Train_temp(:,1:number_inputs)', [1, number_inputs, training_num]);
        Test.inputs = reshape(Test_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
        Train.outputs = reshape(Train_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);
        Test.outputs = reshape(Test_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);
        Train.inputs = dlarray(Train.inputs);
        Train.outputs = dlarray(Train.outputs);
        XTrainA = dlarray(reshape(XTrainA, [1, nF, training_num]));

        if gauss2MF
            [Learnable_parameters] = initialize_gauss2mf(Train.inputs, Train.outputs, number_of_rules, output_membership_type, XTrainA);            
        else
            Learnable_parameters = initialize_Glorot_Kmeans(Train.inputs, Train.outputs, number_of_rules ,output_membership_type, XTrainA);
        end

    elseif dr_method ~= "none"
        [~, WA0] = pca_init(Train_temp(:,1:number_inputs), nF);
        Learnable_parameters.WA0 = WA0;
        XTrainA = cdr_layer(permute(Train_temp(:,1:number_inputs), [3 2 1 ]), Learnable_parameters, number_inputs,-619, nF, dr_method);
        XTrainA = reshape(XTrainA, [1, nF, training_num]);
        Train.inputs = reshape(Train_temp(:,1:number_inputs)', [1, number_inputs, training_num]);
        Test.inputs = reshape(Test_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
        Train.outputs = reshape(Train_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);
        Test.outputs = reshape(Test_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);
        Train.inputs = dlarray(Train.inputs);
        Train.outputs = dlarray(Train.outputs);
        XTrainA = dlarray(XTrainA);

        %% initializing
        if dr_method == "cdr"
            if gauss2MF
                [Learnable_parameters] = initialize_gauss2mf(XTrainA, Train.outputs, number_of_rules, output_membership_type, XTrainA);
            else
                Learnable_parameters = initialize_Glorot_Kmeans(XTrainA, Train.outputs, number_of_rules ,output_membership_type, XTrainA);
            end
            Learnable_parameters.WA0 = dlarray(WA0);

        elseif dr_method == "dr"
            if gauss2MF
                [Learnable_parameters] = initialize_gauss2mf(Train.inputs, Train.outputs, number_of_rules, output_membership_type, XTrainA);
            else
                Learnable_parameters = initialize_Glorot_Kmeans(Train.inputs, Train.outputs, number_of_rules ,output_membership_type, XTrainA);
            end
            Learnable_parameters.WA0 = dlarray(WA0);
        end

    end