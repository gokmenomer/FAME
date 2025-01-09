function [Train, Test] = split_data(data, fracTrain, number_inputs, number_outputs)

    data_size = height(data);
    training_num = round(data_size*fracTrain);
    test_num = data_size - training_num;

    idx = randperm(data_size);

    Training_temp = data(idx(1:training_num),:);
    Testing_temp = data(idx(training_num + 1:end),:);

    %training data
    Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]);
    Train.outputs = reshape(Training_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);

    Train.inputs = dlarray(Train.inputs);
    Train.outputs = dlarray(Train.outputs);

    %testing data
    Test.inputs = reshape(Testing_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
    Test.outputs = reshape(Testing_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);

end