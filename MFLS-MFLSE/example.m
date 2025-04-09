clc;clear;
close all;
%% Possible Options
nF_list  = [2,4,8]; % D for PL
dr_method_list = ["cdr", "dr", "none"]; % cdr | dr for PL
loss_type_list = ["l_f", "l2"];
dataset_name_list = ["BH", "parkinson", "wine", "aids", "concrete","abalone"];
rules = [5, 10];
gauss2MF_list = [true, false]; % true for MFLSE, false for MFLS




%% dataset location
current_path = pwd;

%% Experiment Parameters
nF = 8; % D for PL
dr_method = "cdr";
loss_type = "l2";
gauss2MF = true;
dataset_name = "BH";
seed = 0;
lambda = 0.05;
rng(seed)
dataset_loc = current_path;

[x, y, mbs, learnRate, number_of_epoch] = load_data(dataset_loc,current_path, dataset_name);

    if dr_method == "none" && nF ~= 2
        fprintf("nF should be 2 for no dimensionality reduction\n")
    end
    if dr_method == "none" && loss_type == "l_f"
        fprintf("l_f loss is not supported for no dimensionality reduction\n")
    end


%% configuration of T1-FLS
number_of_rules = 5;

output_membership_type = "linear";

tnorm = "eHTSK";
%% Adam parameters
gradDecay = 0.9;
sqGradDecay = 0.999;
averageGrad = [];
averageSqGrad = [];
%% plotting frequency
plotFrequency = 10;
%% dataset seperation proportions
fracTrain = 0.7;
fracTest = 0.3;
%%

number_inputs = size(x,2);
number_outputs = size(y,2);

%% Normalization upfront
%
[xn,input_mean,input_std] = zscore_norm(x);
[yn,output_mean,output_std] = zscore_norm(y);

data = [xn yn];

%% seperating data

data_size = height(data);
training_num = round(data_size*fracTrain);
test_num = data_size - (training_num);

idx = randperm(data_size);

Training_temp = data(idx(1:training_num),:);
Testing_temp = data(idx(training_num+1:end),:);

%% Input Clustering

[Train, Test, XTrainA, Learnable_parameters, nF] = dr(Training_temp, Testing_temp, dr_method, nF, number_inputs, number_outputs, training_num, test_num, number_of_rules, output_membership_type, gauss2MF);
prev_learnable_parameters = Learnable_parameters;

%% seed reset
rng(seed)

%% reshaping for plotting
yTrue_train = reshape(Train.outputs,[number_outputs, training_num]);
yTrue_test = reshape(Test.outputs,[number_outputs, test_num]);
%% Training loop

number_of_iter_per_epoch = floorDiv(training_num, mbs);
number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;
batch_sum_list = [];
for epoch = 1: number_of_epoch

    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, training_num);

    batch_loss_sum = 0;
    for iter = 1:number_of_iter_per_epoch

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs);

        %calculating loss and gradient
        [loss, gradients, yPred_train] = dlfeval(@fismodelLoss, mini_batch_inputs, number_inputs, targets,number_outputs, number_of_rules, mbs, Learnable_parameters, output_membership_type,tnorm, nF, dr_method, loss_type, lambda);
        batch_loss_sum = batch_loss_sum + loss;
        % updating parameters
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            iter, learnRate, gradDecay, sqGradDecay);


        global_iteration = global_iteration + 1;

    end
    loss = batch_loss_sum / number_of_iter_per_epoch;
    batch_sum_list = [batch_sum_list, loss];

    %testing and plotting
    yPred_test = fismodel(Test.inputs, number_of_rules, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,tnorm, nF, dr_method);
    yPred_test = reshape(yPred_test, [number_outputs, size(Test.inputs,3)]);
    plotter(epoch,plotFrequency,loss,yTrue_test, yPred_test);


end

%% Inference

[train_RMSE, test_RMSE, firing_train, firing_test] = inference(Train, Test, yTrue_train, yTrue_test, ...
                            number_of_rules, number_inputs, number_outputs, Learnable_parameters, ...
                            output_membership_type, tnorm, nF, dr_method, 1);
