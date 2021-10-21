%  Modified from
%  'Made180801_100rep_TrainSetcvglmnet_PredSetcvglmnetPredict_CODE5.m'

clear

%-------------Adjust-------------------------------------------------------
load('example_dataset.mat');

TrainDataset = [];              % matrix of numsubjects x numfeatures
PredictionDataset = [];         % matrix of numsubjects x numfeatures
TrainLabel       = [];          % vector of numsubjects x 1 (containing 0 and 1)
PredictionLabel  = [];          % vector of numsubjects x 1 (containing 0 and 1)

nfolds = 10; 
options = glmnetSet; options.standardize = 1;
options.alpha = 0.5;   % 0 = L1; 1 = L2 

%-------------------------------------------------------------------------

numTrainSubjects = size(TrainDataset,1); 

CVIdx = crossvalind('Kfold', numTrainSubjects, nfolds);

% Train:
CVerr = cvglmnet(TrainDataset, TrainLabel,'binomial', options, 'auc', nfolds, CVIdx, false, true, true);
                 % x, y, family, options, type, nfolds, goldid, parallel, keep, grouped                 

% Save model parameters:
ModelParam = [CVerr.glmnet_fit.a0(find(CVerr.lambda == CVerr.lambda_min)); ...
              CVerr.glmnet_fit.beta(:,find(CVerr.lambda == CVerr.lambda_min))];
          
% Predict:
Logits = cvglmnetPredict(CVerr, PredictionDataset, CVerr.lambda_min, 'link');
                         % object, newx, s, type

% Calculate AUC:
[X,Y,Threshold,AUC] = perfcurve(PredictionLabel, Logits, 1);

figure; plot(X,Y)


