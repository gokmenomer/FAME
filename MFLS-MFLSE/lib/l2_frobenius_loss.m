function loss = l2_frobenius_loss(yPred, targets, learnable_parameters, lambda, type)

    % Types: "l2" or "l_f"
    if type == "l2"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        loss = l2_loss;
    elseif type == "l_f"
        l2_loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
        frobenius_term = sum(sum(learnable_parameters.WA0(1:end-1,1:end).^2));  % Squared Frobenius norm
        loss = l2_loss + (lambda/2)*frobenius_term ;
    end
end
