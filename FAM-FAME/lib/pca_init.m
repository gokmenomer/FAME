function [XTrainA, WA0] = pca_init( XTrain, nF)

    [WPCA, XTrainA] = pca(XTrain, "NumComponents",nF);
    [WA0, WC0] = deal([ WPCA ; zeros(1, nF)]);
    [MA, MC] = deal(nF);

end