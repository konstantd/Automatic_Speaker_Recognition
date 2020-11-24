tic

%% Clearing
clear all;
close all;
clc;


%% Reading.
load training_set.mat;
data = AF_sum;
Classes = 10;

% Proof that data contains same samples from each class.
count = zeros(1, Classes);
for r = 1 : length(data)
    count(data(r, end)) = count(data(r, end)) + 1;
end


%% Skipping same rows.
temp_data = unique(data,'rows');
data = temp_data;


%% Normalizing.
for i = 1 : size(data,2) - 1
    temp = rescale(data(:,i));
    data(:,i) = temp;
end


%% Skipping NaN columns.
out = data(:,all(~isnan(data)));   % for nan - columns
data = out;



%% Choosing Features.
X = data(:, 1:end-1); %Columns have not been shuffled.
y = data(:, end);
k = 200; % Nearest Neighbors
[idx, weights] = relieff(X, y, k);



%% Shuffling Data.
shuffledData = zeros(size(data));
shuffledIndex = randperm(length(data)); % Array of random positions.

for r = 1:length(data)
    shuffledData(r, :) = data(shuffledIndex(r), :);
end
data = shuffledData; 



%% Splitting.
N = length(data);
trainingData = data(1 : round(N*0.7) , :);
checkData = data(round(N*0.7) + 1 : end , :);



%% Accuracies for all Possible NFs.
Accuracy = zeros(1,size(data,2)-1);

for NF = 1: size(data,2)-1
    
    trainingData_x = trainingData(:, idx(1:NF));
    trainingData_y = trainingData(:, end);

    checkData_x = checkData(:, idx(1:NF));
    checkData_y = checkData(:, end);
    
  
    %% KNN Model.
    Mdl = fitcknn(trainingData_x, trainingData_y, 'NumNeighbors', 3);
    rloss = resubLoss(Mdl);

    %CVMdl = crossval(Mdl);
    %kloss = kfoldLoss(CVMdl);


    %% Predict.
    output = predict(Mdl, checkData_x);
    ConfusionMatrix = confusionmat(checkData_y, output,'Order', [1 2 3 4 5 6 7 8 9 10]);

    for i = 1:size(ConfusionMatrix,2)
        Accuracy(NF) = Accuracy(NF) + ConfusionMatrix(i,i);
    end

    Accuracy(NF) = Accuracy(NF)/length(checkData);

end

figure;
plot(Accuracy)
xlabel('Number of Features');
ylabel('Accuracy');
saveas(gcf,'Relief_kNN/Accuracy~Features.png')

[max_Accuracy, max_NF] = max(Accuracy);




%% Training with best NF.
data_x = data(:, idx(1:max_NF));
data_y = data(:, end);

Mdl = fitcknn(data_x, data_y, 'NumNeighbors', 3);
rloss = resubLoss(Mdl);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Testing

Overall_Accuracy = 0;
testData = cell(1,Classes);
ConfusionMatrix = cell(1,Classes);

for sp = 1:Classes
    
    %% Import test set for speaker i.
    % edw 8a bei kapoia cd gia subfolderTestSets
    str = sprintf('Speaker %d.mat',sp);
    load(str);
	testData{sp} = AF_test;

    % Normalizing.
    for i = 1 : size(testData{sp},2) - 1
        temp = rescale(testData{sp}(:,i));
        testData{sp}(:,i) = temp;
    end

    out = testData{sp}(:,all(~isnan(testData{sp})));  
    testData{sp} = out;

    testData_x = testData{sp}(:, idx(1:max_NF));
    testData_y = testData{sp}(:, end);


    %% Make Prediction.
    output = predict(Mdl, testData_x);
    ConfusionMatrix{sp} = confusionmat(testData_y, output, 'Order', [1 2 3 4 5 6 7 8 9 10]);

    Possibility = 0;

    for i = 1:size(ConfusionMatrix{sp},2)
        Possibility = Possibility + ConfusionMatrix{sp}(i,i);
    end

    Possibility = Possibility/length(testData_x);

    Predictions = ConfusionMatrix{sp}(sp,:);

    [samplesFound, Speaker] = max(Predictions);

    fprintf('\nSpeaker %d with probability %.2f %% \n', Speaker, 100*Possibility);
    
    if (Speaker == testData_y(1))
        
        Overall_Accuracy = Overall_Accuracy + 1;
    end
    

end

Overall_Accuracy = Overall_Accuracy/10;
fprintf('\nOverall Accucarcy %.2f %% \n\n', 100*Overall_Accuracy);

toc



% Speaker 9 with probability 19.11 % 
% 
% Speaker 9 with probability 14.92 % 
% 
% Speaker 3 with probability 20.86 % 
% 
% Speaker 4 with probability 23.49 % 
% 
% Speaker 4 with probability 21.08 % 
% 
% Speaker 1 with probability 2.35 % 
% 
% Speaker 4 with probability 6.60 % 
% 
% Speaker 4 with probability 16.58 % 
% 
% Speaker 9 with probability 36.85 % 
% 
% Speaker 9 with probability 7.71 % 
% 
% Overall Accucarcy 30.00 % 
% 
% Elapsed time is 3869.674381 seconds.
