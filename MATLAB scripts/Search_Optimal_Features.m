% The following code solves the problem of Feature Selection for a Dataset
% of 10 Speakers. Relief algorithm is used to assign weights to each
% feature. Then we train differnet Multi-Layer Perceptrons with variable
% number of input features. Cross-Entropy is used to calculate the optimal
% model that has been trained. The results are saved and loaded in
% Neural_Network_with_selected_Features.

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Clearing
clear all;
close all;
clc;



%% Starting.
fprintf('*****  %s  *****\n', mfilename);



%% Reading
load training_set.mat;
data = AF_sum;
Classes = 10;



%% Skipping same rows
temp_data = unique(data,'rows');
data = temp_data;



%% Choosing Features
X = data(:, 1:end-1); 
y = data(:, end);
k = 10; 
[idx, weights] = relieff(X, y, k);
% Save weights of features
save('idx','idx')
save('weights','weights')



%% Shuffling Data
rng(0);
shuffledData = zeros(size(data));
shuffledIndex = randperm(length(data)); % Array of random positions.

for r = 1:length(data)
    
    shuffledData(r, :) = data(shuffledIndex(r), :);
    
end

data = shuffledData; 


%% Finding optimal number of features based on Cross-Entropy
cross_entropy = zeros(12, 1);
data = transpose(data);
save('data','data');


%% Compatible form of Data
myInputs = data(1:end-1, :);
myTarget = zeros(Classes, size(data,2));

for  i =1:size(data,2)
    
    myTarget(data(end,i) , i) = 1;
    
end


%%  Finding minimum Cross- Entropy
for FeaturesNumber = 10: 10: 120
      
    fprintf(" Training Model with %d Features. \n", FeaturesNumber);
    
    %% k-Fold Cross Validation    
    c = cvpartition(data(end,:), 'KFold', 5);
        
    for i = 1:c.NumTestSets
            
        trainingIDs = c.training(i);
        validationIDs = c.test(i);
            
        trainingData_x = myInputs(idx(1:FeaturesNumber), trainingIDs);
        trainingData_y = myTarget(:, trainingIDs);

        validationData_x = myInputs(idx(1:FeaturesNumber), validationIDs);
        validationData_y = myTarget(:, validationIDs);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Neural Network
        rng(0);   % initialize the RNG to the same state before training to obtain reproducibility
    
        x = trainingData_x;
        t = trainingData_y;

        % Choose a Training Function
        % For a list of all training functions type: help nntrain
        % 'trainlm' is usually fastest.
        % 'trainbr' takes longer but may be better for challenging problems.
        % 'trainscg' uses less memory. Suitable in low memory situations.
        trainFcn = 'trainrp';  % Scaled conjugate gradient backpropagation.
        performFcn = 'crossentropy';

        % Create a Pattern Recognition Network
        hiddenLayerSize = [20 80];
        net = patternnet(hiddenLayerSize, trainFcn, performFcn);

        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;

        % Train the Network
        net = configure(net,x,t);  % remove previous weights and reinitialize with random weights.
        [net, tr] = train(net,x,t);

        % Test the Network
        y = net(validationData_x);
        performance = perform(net,validationData_y, y);

        cross_entropy(FeaturesNumber/10) = cross_entropy(FeaturesNumber/10) + performance;

    end
    
    cross_entropy(FeaturesNumber/10) = cross_entropy(FeaturesNumber/10) / c.NumTestSets ;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1);
plot(cross_entropy)
xlabel('Number of Features (x10)');
ylabel('Cross Entropy');
saveas(gcf,'CVMLP/MSE~Number_of_Features.png')

[minCrossEntropy, optimalNF] = min(cross_entropy);
optimalNF = optimalNF * 10;
fprintf("\nThe optimal number of features is %d.\n\n", optimalNF);

toc

% The optimal number of features is 50.
% Elapsed time is 2689.509649 seconds.
