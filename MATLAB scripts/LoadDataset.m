clear all;
close all;
clc;

load training_set.mat
data = AF_sum;

load (idx.mat);
NEWdata = data(:,idx(1:50));
NEWdata(:,end+1) = data(:,end);


% for i = 1:size(data,2)-1
%     
%     data_min = min(data(:,i));
%     data_max = max(data(:,i));
%     
%     data(:,i) = ( data(:,i) - data_min ) / ( data_max - data_min );
%     
% end
% 
% 
% 
% data_KNN = data(:,idx(1:50));
% data_KNN(:,end+1) = data(:,end);
