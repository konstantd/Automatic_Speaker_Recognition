% Training a Cubic SVM model that was exported from Classidication Learner 
% Toolbox. This model was trained with the 50 most importnant features of 
% Relief Algorithm. 

tic


%% Clearing
clear all;
close all;
clc;



%% Starting.
fprintf('*****  %s  *****\n', mfilename);



%% Load Values calculated in Search_Optimal_Features.m
Classes = 10;
optimalNF = 50;
load CubicSVM.mat;
load idx.mat;



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


    %% Make Prediction     
    yfit = CubicSVM.predictFcn(testData_x);
    
    
    %% Confusion Matrix 
    ConfusionMatrix{speaker} = confusionmat(testData_y, yfit, 'Order', [1 2 3 4 5 6 7 8 9 10]);
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



%% Overall Accuracy
fprintf('\n\nModel predicted %d out of 10 speakers correctly.\n', Overall_Accuracy);


%% Bar plot
figure();
bar(Possibility);
xlabel("Speakers");
ylabel("Accuracy %");
saveas(gcf, 'Cubic_SVM/Accuracies.png')

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

% Elapsed time is 5.563175 seconds.