% Modified from: 'Made180823_nestedCV_cvglmnetcvglmnetPredict.m'


clear

%------------------Adjust--------------------------------------------------

load('example_dataset.mat');

repnum = 100; %10 or 100;
foldn = 10;
options = glmnetSet; options.alpha = 0; % 0 = L1; 1 = L2 

Dataset = []; % matrix of numsubjects x numfeatures
                                                     
biDX = []; % vector of numsubjects x 1 (containing 0 and 1)

%--------------------------------------------------------------------------


numsubjects = size(Dataset,1);
numfeat = size(Dataset,2);

% To save over several repetitions of nested CV:
Collection_estimated_total_performances = zeros(1, repnum);
Collection_hyperparameters = zeros(1, repnum);
Collection_CVIdx = zeros(numsubjects, repnum);
Collection_ModelParam = zeros(numfeat+1,repnum);
Collection_TestSets_biDX = zeros(numsubjects,repnum);
Collection_ExternalTestSet_Scores_table = zeros(numsubjects,repnum);% predicted values when 'response', logits when 'link'

    for rep = 1:repnum
        fprintf('\n ====== Repetition %d ======\n', rep)

        CVIdx = crossvalind('Kfold', numsubjects, 10); 
        Collection_CVIdx(:,rep) = CVIdx;
        
        TestSets_biDX = [ biDX(CVIdx==1); biDX(CVIdx==2); biDX(CVIdx==3); biDX(CVIdx==4); biDX(CVIdx==5); biDX(CVIdx==6); biDX(CVIdx==7); biDX(CVIdx==8); biDX(CVIdx==9); biDX(CVIdx==10)];
        Collection_TestSets_biDX(:,rep) = TestSets_biDX;  
          
        % Find which surrogate group has most subjects(longest length):
         testdatalength(1) = length(Dataset(CVIdx==1));
         testdatalength(2) = length(Dataset(CVIdx==2));
         testdatalength(3) = length(Dataset(CVIdx==3));
         testdatalength(4) = length(Dataset(CVIdx==4));
         testdatalength(5) = length(Dataset(CVIdx==5));
         testdatalength(6) = length(Dataset(CVIdx==6));
         testdatalength(7) = length(Dataset(CVIdx==7));
         testdatalength(8) = length(Dataset(CVIdx==8));
         testdatalength(9) = length(Dataset(CVIdx==9));
         testdatalength(10) = length(Dataset(CVIdx==10)); 
         max_testdatalength = max(testdatalength);
       
        % To save from each of 10 external folds: 
        Xcoordinates = zeros(max_testdatalength, foldn);
        Ycoordinates = zeros(max_testdatalength, foldn);
        AUC_values = zeros(1, foldn);
        Hyperparameters = struct('lambda_fold1', NaN, 'lambda_fold2', NaN, ...
            'lambda_fold3', NaN, 'lambda_fold4', NaN, ...
            'lambda_fold5', NaN, 'lambda_fold6', NaN, ...
            'lambda_fold7', NaN, 'lambda_fold8', NaN, ...
            'lambda_fold9', NaN, 'lambda_fold10', NaN);
            fieldH = fieldnames(Hyperparameters);
        
        % To save scores of one run of nested CV:
        ExternalTestSet_Scores = struct('scores_fold1', NaN, 'scores_fold2', NaN, ...
                                'scores_fold3', NaN, 'scores_fold4', NaN, ...
                                'scores_fold5', NaN, 'scores_fold6', NaN, ...
                                'scores_fold7', NaN, 'scores_fold8', NaN, ...
                                'scores_fold9', NaN, 'scores_fold10', NaN);
                                fieldC = fieldnames(ExternalTestSet_Scores);
        ExternalTestSet_Scores_table = [];
        
        
        
        %% Nested CV starts here.
        for k = 1:foldn
            fprintf('\n*******Doing External fold %d of %d*******\n',k,foldn)
            
            % Define External Folds (train set and test)
            extTrainData  = Dataset(CVIdx~=k, :);
            extTrainLabel = biDX(CVIdx~=k);
            extTestData  = Dataset(CVIdx==k, :);
            extTestLabel = biDX(CVIdx==k);
            currenttestdatalength = size(extTestData,1);
            
            % CV on external TrainData to find best lambda (lambda_min):
            k2Idx = crossvalind('Kfold', length(extTrainData), foldn);
            CVerr = cvglmnet(extTrainData, extTrainLabel, 'binomial', options, 'auc', foldn, k2Idx, false, true, true);
            Hyperparameters.(fieldH{k}) = [CVerr.lambda_min];
            Collection_ModelParam(:,k) = [CVerr.glmnet_fit.a0(find(CVerr.lambda==CVerr.lambda_min)); CVerr.glmnet_fit.beta(:,find(CVerr.lambda==CVerr.lambda_min)) ];
            
            
            % Score each external test set 
            % using model parameters obtained from external TrainData 
            % with best lambda from internal loop, output as logit values: 
            Scores = cvglmnetPredict(CVerr, extTestData, CVerr.lambda_min, 'link');
            ExternalTestSet_Scores.(fieldC{k})(1:testdatalength(k),rep) = Scores;
            ExternalTestSet_Scores_table = [ExternalTestSet_Scores_table;Scores];
            [X,Y,T,AUC] = perfcurve(extTestLabel, Scores, 1);
            AUC_values(1,k) = AUC; 
            
            % Save plot coordinates for performance on external test set:
            if length(X) ~= 2
                Xcoordinates(1:currenttestdatalength+1, k) = X;
                Ycoordinates(1:currenttestdatalength+1, k) = Y;
            elseif length(X) == 2
                Xcoordinates(1:2, k) = X;
                Ycoordinates(1:2, k) = Y;
            end
            
        end
        %% --One run of nested CV ends here---
        
        % Fix any bugs in coordinates:
        Xcoordinates(end, find(Xcoordinates(end,:)==0)) = 1;
        Ycoordinates(end, find(Ycoordinates(end,:)==0)) = 1;
        % Average the coordinate values of each external test set:
        Collection_of_meanXcoordinates(1:length(Xcoordinates), rep) = mean(Xcoordinates, 2);
        Collection_of_meanYcoordinates(1:length(Ycoordinates), rep) = mean(Ycoordinates, 2);
      
        % Plot the average coordinates of one run of nested CV:
        plot(Collection_of_meanXcoordinates(:,rep), Collection_of_meanYcoordinates(:,rep))
        hold on
        
        % Average the AUCs of each external test set, and save:
        Avg_estimated_performance = mean(AUC_values)
        Collection_estimated_total_performances(1,rep) = Avg_estimated_performance;
        
        % Average the best lambda values from each external test set, and save:
        Avg_Hyperparameter = ( Hyperparameters.(fieldH{1})+ Hyperparameters.(fieldH{2})+ Hyperparameters.(fieldH{3})+ ...
                               Hyperparameters.(fieldH{4})+ Hyperparameters.(fieldH{5})+ Hyperparameters.(fieldH{6})+ ...
                               Hyperparameters.(fieldH{7})+ Hyperparameters.(fieldH{8})+ Hyperparameters.(fieldH{9})+ ...
                               Hyperparameters.(fieldH{10}))/foldn  
        Collection_hyperparameters(1,rep) = Avg_Hyperparameter;
        Collection_ExternalTestSet_Scores_table(:,rep) = ExternalTestSet_Scores_table;
        
        
    end
    
    % Average of coordinates from each run of nested CV 
    % (THE FINAL AVERAGE COORDINATES):
    Final100rep_Xcoordinates = mean(Collection_of_meanXcoordinates, 2);
    Final100rep_Ycoordinates = mean(Collection_of_meanYcoordinates, 2);
    plot(Final100rep_Xcoordinates, Final100rep_Ycoordinates)
    hold off
    % Plot THE FINAL AVERAGE AUC OF MANY RUNS OF NESTED CV:
    figure
    plot(Final100rep_Xcoordinates, Final100rep_Ycoordinates)
    FinalAUC = mean(Collection_estimated_total_performances)

   
    ExternalTestSetScores_aligned = [ExternalTestSet_Scores.scores_fold1; ExternalTestSet_Scores.scores_fold2; ExternalTestSet_Scores.scores_fold3; ExternalTestSet_Scores.scores_fold4; ...
        ExternalTestSet_Scores.scores_fold5; ExternalTestSet_Scores.scores_fold6; ExternalTestSet_Scores.scores_fold7; ExternalTestSet_Scores.scores_fold8; ...
        ExternalTestSet_Scores.scores_fold9; ExternalTestSet_Scores.scores_fold10; ];