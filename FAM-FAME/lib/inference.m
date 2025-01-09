function [TrainLoss, valLoss, TestLoss, train_RMSE, val_RMSE, test_RMSE, yPred_Train, yPred_Val, yPred_Test] = inference(Train, Val, Test,  number_inputs, number_outputs, number_of_rules, Learnable_parameters, output_membership_type, tnorm, nF, dr_method, loss_type, gam_aggregration_method, lambda)
    %% Inference
    [TrainLoss, yPred_Train] = evaluate(Train.inputs, number_inputs, Train.outputs, number_outputs, number_of_rules, length(Train.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method, loss_type, gam_aggregration_method, lambda);
    [valLoss ,yPred_Val] = evaluate(Val.inputs, number_inputs, Val.outputs, number_outputs, number_of_rules, length(Val.inputs), Learnable_parameters, output_membership_type, tnorm, nF, dr_method, loss_type, gam_aggregration_method, lambda);
    [TestLoss ,yPred_Test] = evaluate(Test.inputs, number_inputs, Test.outputs, number_outputs, number_of_rules, length(Test.inputs), Learnable_parameters, output_membership_type, tnorm, nF, dr_method, loss_type, gam_aggregration_method, lambda);

    train_RMSE = rmse(yPred_Train, Train.outputs);
    val_RMSE = rmse(yPred_Val, Val.outputs);
    test_RMSE = rmse(yPred_Test, Test.outputs);
end