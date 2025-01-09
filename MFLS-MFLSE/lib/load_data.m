function [x, y, mbs, learnRate, number_of_epoch] = load_data(dataset_loc, current_path, dataset_name)
    if dataset_name == "BH"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("BH.mat");
        cd(current_path);
        data = [x y];
        mbs = 64;
        learnRate = 0.01;
        number_of_epoch = 100;
    elseif dataset_name == "abalone"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("abalone.mat");
        cd(current_path);
        data = [x y];
        mbs = 64;
        learnRate = 0.01;
        number_of_epoch = 100;
    elseif dataset_name == "concrete"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("Concrete.mat");
        cd(current_path);
        data = [x y];
        mbs = 64;
        learnRate = 0.01;
        number_of_epoch = 100;
    elseif dataset_name == "parkinson"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("parkinson.mat");
        cd(current_path);
        y = y_motor;
        data = [x y];
        mbs = 512;
        learnRate = 0.01;
        number_of_epoch = 1000;
    elseif dataset_name == "aids"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("aids.mat");
        cd(current_path);
        data = [x y];
        mbs = 64;
        learnRate = 0.01;
        number_of_epoch = 100;
    elseif dataset_name == "wine"
        cd(dataset_loc)
        cd("UCI_datasets")
        load("whitewine.mat");
        cd(current_path);
        data = [x y];
        mbs = 64;
        learnRate = 0.01;
        number_of_epoch = 100;
    end

end