% Training a Neural Network with optimal Number of Features, as this was
% calculated by Search_Optimal_Features. In this script 5 Models are
% trained using cvpartition, which decide for the output.

tic



%% Clearing
clear all;
close all;
clc;


%% Starting.
fprintf('*****  %s  *****\n', mfilename);


%% Load Values calculated in Search_Optimal_Features.m
optimalNF = 47;
Classes = 10;
load training_set.mat;
data = AF_sum;
load('idx.mat')



%% Shuffling Data.
rng(0);
shuffledData = zeros(size(data));
shuffledIndex = randperm(length(data)); % Array of random positions.

for r = 1:length(data)
    
    shuffledData(r, :) = data(shuffledIndex(r), :);
    
end

data = shuffledData; 
data = transpose(data);



%% Transforming Inputs and Outputs to compatible form
myInputs = data(idx(1:optimalNF), :);
myTarget = zeros(Classes, size(data,2));

for  i =1:size(data,2)
    
    myTarget(data(end,i), i) = 1;
    
end
    

%% k-Fold Cross Validation    
c = cvpartition(data(end,:), 'KFold', 5);
net = cell(1,5);
        
for i = 1:c.NumTestSets
            
    trainingIDs = c.training(i);
    validationIDs = c.test(i);
            
    trainingData_x = myInputs(:, trainingIDs);
    trainingData_y = myTarget(:, trainingIDs);

    validationData_x = myInputs(:, validationIDs);
	validationData_y = myTarget(:, validationIDs);
    
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Neural Network
	rng(0);   % initialize the RNG to the same state before training to obtain reproducibility
	x = trainingData_x;
	t = trainingData_y;
    
	% Create a Pattern Recognition Network
	hiddenLayerSize = [20 80];
    trainFcn = 'trainrp';
    performFcn = 'crossentropy';
	net{i} = patternnet(hiddenLayerSize, trainFcn, performFcn);

	% Setup Division of Data for Training, Validation, Testing
	net{i}.divideParam.trainRatio = 70/100;
	net{i}.divideParam.valRatio = 15/100;
	net{i}.divideParam.testRatio = 15/100;

	% Train the Network
	net{i} = configure(net{i},x,t);  %remove previous weights and reinitialize with random weights.
	[net{i}, tr] = train(net{i},x,t);

	% Test the Network
	y = net{i}(validationData_x);


end
    


%% Testing the Models
testData = cell(1, Classes);
ConfusionMatrix = cell(1, Classes);
Possibility = zeros(1, Classes);
Overall_Accuracy = 0;


%% Loading Datasets for each Speaker
for speaker = 1:Classes
    
    %% Import test set for speaker i.
    str = sprintf('Speaker %d.mat', speaker);
    load(str);
	testData{speaker} = AF_test;

    testData_x = testData{speaker}(:, idx(1:optimalNF));
    testData_y = testData{speaker}(:, end);

    testData_x = transpose(testData_x);  

    %% Make Prediction for 5 Neural Nets
    output = zeros(Classes, length(testData_y));
    
    for i = 1:5
        
        output = output + 0.2 * round(net{i}(testData_x));
        
    end
    
    % Finding which class is most predicted for each frame
    position = zeros(1,size(output,2));
    
    for i = 1:size(output,2)
        
        [maximum, position(i)] = max(output(:,i));
        
    end
    
    % Assigning to this class 1 and 0 to the rest
    output = zeros(size(output));
    
    for i = 1:size(output,2)
        
        output(position(i), i) = 1;
        
    end
    
    % Transdorming Output to compatible form of a vector
    output2 = zeros(size(output,2),1);
    
    for  i =1:size(output,2)
    
        output2(i) = 1*output(1,i) + 2*output(2,i) + 3*output(3,i) + 4*output(4,i) + 5*output(5,i) + 6*output(6,i) + 7*output(7,i) + 8*output(8,i) + 9*output(9,i) + 10*output(10,i);
        
        % Boundary Conditions
        
        if (output2(i) < 1)
            output2(i) = 1;
            
        elseif (output2(i) > Classes)
            output2(i) = Classes;
            
        end
     
    end

    
    %% Confusion Matrix 
    ConfusionMatrix{speaker} = confusionmat(testData_y, output2, 'Order', [1 2 3 4 5 6 7 8 9 10]);
    Possibility(speaker) = sum(diag(ConfusionMatrix{speaker}));   
    Possibility(speaker) = 100*Possibility(speaker)/length(testData_y);
    Predictions = ConfusionMatrix{speaker}(speaker,:);
    
    [samplesFound, PredictedSpeaker] = max(Predictions);

    fprintf('\nSpeaker %d with probability %.2f %%', PredictedSpeaker, Possibility(speaker));
    
    if (PredictedSpeaker == testData_y(1))
        
        Overall_Accuracy = Overall_Accuracy + 1;
        
    end
    
end


%% Overall Confusion Matrix
OverallConfusionMatrix = zeros(Classes);

for i = 1:Classes
    
   OverallConfusionMatrix(i,:) =  ConfusionMatrix{i}(i,:);
    
end

% OverallConfusionMatrix
save('OverallConfusionMatrix', 'OverallConfusionMatrix');


%% Overall Accuracy
fprintf('\n\nModel predicted %d out of 10 speakers correctly.\n', Overall_Accuracy);


%% Bar plot
figure();
bar(Possibility);
xlabel("Speakers");
ylabel("Accuracy %");
saveas(gcf, 'CVMLP/Accuracies.png')

fprintf("\nAverage Accuracy: %.2f %%", mean(Possibility));


%% Recall
recall = zeros(1,size(OverallConfusionMatrix, 1));

for i =1:size(OverallConfusionMatrix,1)
    
    recall(i) = OverallConfusionMatrix(i,i)/sum(OverallConfusionMatrix(i,:));
    
end

Recall = sum(recall)/size(OverallConfusionMatrix,1);
fprintf("\nRecall: %.2f %%", Recall*100);


%% Precision
precision = zeros(1,size(OverallConfusionMatrix,1));

for i =1:size(OverallConfusionMatrix,1)
    
    precision(i)=OverallConfusionMatrix(i,i)/sum(OverallConfusionMatrix(:,i));
    
end

Precision = sum(precision)/size(OverallConfusionMatrix,1);
fprintf("\nPrecision: %.2f %%", Precision*100);


%% F score
F_score = 2*Recall*Precision/(Precision+Recall);
fprintf("\nF_Score: %.2f %% \n\n", F_score*100);


toc

% Elapsed time is 50.299990 seconds.