clc;clear;
close all;
%% Possible Options
nF_list  = [2,4,8]; % D for PL
dr_method_list = ["cdr", "none"]; % cdr for PL
loss_type_list = ["l_f", "l2"];
dataset_name_list = ["BH", "parkinson", "wine", "aids", "concrete","abalone"];
rules = [5, 10];
gam_aggregration_method_list = ["sum", "weighted"];
gauss2MF_list = [true, false]; % true for FAME, false for FAM


%% dataset location
current_path = pwd;
dataset_loc = current_path;

%% Experiment Parameters
nF = 4; % D for PL
dr_method = "cdr";
loss_type = "l2";
gam_aggregration_method = "sum";
gauss2MF = false;
dataset_name = "concrete";
seed = 0;
lambda = 0.05;
rng(seed)

[x, y, mbs, learnRate, number_of_epoch] = load_data(dataset_loc, current_path, dataset_name);

%% configuration of T1-FLS
number_of_rules = 5;

output_membership_type = "linear";

tnorm = "product";
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

number_inputs = size(x,2);
number_outputs = size(y,2);

%% Normalization upfront

[xn, x_minimum, x_range] = zscore_norm(x);
[yn, y_minimum, y_range] = zscore_norm(y);

data = [xn yn];

%% seperating data

data_size = height(data);

training_num = round(data_size*fracTrain);
test_num = data_size - training_num;

[Train, Test] = split_data(data, fracTrain, number_inputs, number_outputs);
Val = Test;

%% initializing

Learnable_parameters = initialize_model(Train.inputs, Train.outputs, number_of_rules ,output_membership_type, nF, dr_method, gauss2MF);
prev_learnable_parameters = Learnable_parameters;

%% seed reset
rng(seed)

%% reshaping for plotting
yTrue_train = reshape(Train.outputs,[number_outputs, training_num]);
yTrue_val = reshape(Val.outputs,[number_outputs, test_num]);
yTrue_test = reshape(Test.outputs,[number_outputs, test_num]);

%% Valstep

minValLoss = inf;
minValLossInt = inf;
minValLossTune = inf;
estCounter = 0;
estCounterInt = 0; 
estCounterTune = 0;

earlyStopThreshold = [50];


%% Training loop

number_of_iter_per_epoch = floorDiv(training_num, mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

for epoch = 1: number_of_epoch

    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, training_num);

    batch_loss = 0;

    for iter = 1:number_of_iter_per_epoch

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets, iter, mbs);

        %calculating loss and gradient
        [loss, gradients, ~] = dlfeval(@fismodelLoss, mini_batch_inputs, number_inputs, targets,number_outputs, number_of_rules, mbs, Learnable_parameters, output_membership_type,tnorm,  nF, dr_method, loss_type, gam_aggregration_method, lambda);

        % updating parameters
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            epoch, learnRate, gradDecay, sqGradDecay);

        batch_loss = batch_loss + loss;

    end

    batch_loss = batch_loss/number_of_iter_per_epoch;


    %Validaiton and plotting
    [TrainLoss, ~] = evaluate(Train.inputs, number_inputs, Train.outputs, number_outputs, number_of_rules, length(Train.inputs), Learnable_parameters, output_membership_type,tnorm,  nF, dr_method, loss_type, gam_aggregration_method, lambda);
    [valLoss ,yPred_Val] = evaluate(Val.inputs, number_inputs, Val.outputs, number_outputs, number_of_rules, length(Val.inputs), Learnable_parameters, output_membership_type, tnorm,  nF, dr_method, loss_type, gam_aggregration_method, lambda);

    yPred_Val = reshape(yPred_Val, [number_outputs, size(Val.inputs,3)]);

   if valLoss < minValLoss
       minValLoss = valLoss;
       estCounter = 0;
       bestModel = Learnable_parameters;
   else
       estCounter = estCounter + 1; % Increment patience counter
   end

   fprintf('Main Effect Epoch %d, Train Loss: %.4f, Test Loss: %.4f, estCounter: %d\n', epoch, TrainLoss, valLoss, estCounter);

   plotter(epoch,plotFrequency,batch_loss,yTrue_val, yPred_Val);


end

%% Inference

[TrainLoss, valLoss, TestLoss, train_RMSE, val_RMSE, test_RMSE, yPred_Train, yPred_Val, yPred_Test] = inference(Train, Val, Test, number_inputs, number_outputs, number_of_rules, Learnable_parameters, output_membership_type, tnorm, nF, dr_method, loss_type, gam_aggregration_method, lambda);
fprintf('Train RMSE %4f, Val RMSE %4f, Test RMSE %4f\n', train_RMSE, val_RMSE, test_RMSE);
fprintf('Size of Train %d, Val %d, Test %d\n', length(Train.inputs), length(Val.inputs), length(Test.inputs));
%% Inference for Plotting

plot_index = 1;
results = plot_inference( number_of_rules, number_inputs, number_outputs, length(Train.inputs), Learnable_parameters, output_membership_type, nF, dr_method, plot_index);



