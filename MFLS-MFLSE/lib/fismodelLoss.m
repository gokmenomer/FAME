function [loss, gradients, yPred] = fismodelLoss(mini_batch_inputs, number_inputs, targets, number_outputs, number_of_rules, mbs, learnable_parameters, output_membership_type,tnorm, nF, dr_method, loss_type, lambda)

yPred = fismodel(mini_batch_inputs, number_of_rules, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type,tnorm, nF, dr_method);


type = loss_type;
loss = l2_frobenius_loss(yPred, targets, learnable_parameters, lambda, type);
gradients = dlgradient(loss, learnable_parameters);

end