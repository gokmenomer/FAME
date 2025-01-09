function [XTrainA, WA0] = pca_init( XTrain, nF)

    [WPCA, XTrainA] = pca(XTrain, "NumComponents",nF);
    [WA0, ~] = deal([ WPCA ; zeros(1, nF)]);
end