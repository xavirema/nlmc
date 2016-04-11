%     nlmc implements Non-Linear Matrix Completion
%     Copyright (C) 2016 Xavier Alameda-Pineda [xavi.alameda@gmail.com]
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
%     Please cite the following article if you use this software:
%     
%     X. Alameda-Pineda, E. Ricci, Y. Yan and N. Sebe, 
%     Recognizing Emotions from Abstract Paintings using Non-Linear Matrix Completion
%     IEEE Computer Vision and Pattern Recognition 2016.


%%% Init & Misc
clear; clc;
warning('off');
%%% Get paths
dataPath = '../feature/';
resultsPath = '../results/';
nlmcPath = '../src/';
slepPath = '';
tsvmPath = '';
svmPath = '';

%%% Dataset
dataset = 'MART';
% dataset = 'deviantArt';
if ~exist('dataset','var'),
    error('You must select a dataset.');
end

% Percentages
percs = 10:10:90;
% N random splits
nTrials = 12;

%%% Method
method = 'nlmc';
% method = 'lmc';
% method = 'tsvm'; if isempty(tsvmPath), error('You must install the TSVM package from: http://svmlight.joachims.org/ and set "tsvmPath" to the install directory.'); end
% method = 'svm'; if isempty(svmPath), error('You must install the Lib_SVM package from: http://www.csie.ntu.edu.tw/~cjlin/libsvm/ and set "svmPath" to the install directory.'); end
% method = 'lasso'; if isempty(slepPath), error('You must install the SLEP package from: http://www.yelab.net/software/SLEP/ and set "slepPath" to the install directory.'); end
% method = 'group-lasso'; if isempty(slepPath), error('You must install the SLEP package from: http://www.yelab.net/software/SLEP/ and set "slepPath" to the install directory.'); end
if ~exist('method','var'),
    error('You must select a method.');
end

%%% Parameters
% RBFKernelParam = 10.^(-2:0.5:2);
% RegularizationMC = 10.^(-3:3);
% DecompositionSize = 2:5;
RegularizationTSVM = RegularizationMC;
RegularizationSVM = RegularizationMC;
RegularizationLASSO = RegularizationMC;
RegularizationGroupLASSO = RegularizationMC;

% Load data
tmp = load([dataPath dataset '/' dataset '_allPCAData.mat']);
data = tmp.data;

%%% Loop
for pp = percs,
    fprintf('==================\n');
    fprintf(' Training: %02d%% \n',pp);
    fprintf('==================\n');    
    % Results Name
    resultsName = [dataset '/' dataset '_' num2str(pp) '_' method];
    %%% Switch method
    switch method
        case 'nlmc'
	    % Path
	    addpath(nlmcPath);
            % Parameters
            KP = RBFKernelParam;
            RP = DecompositionSize;
            Lambda = RegularizationMC;
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(Lambda),numel(KP),numel(RP));
            accuracyTestEmotion = zeros(nTrials,numel(Lambda),numel(KP),numel(RP));
            % Output
            fprintf('nlmc === Starting cross-validation with:');
            fprintf('\nLambda = '); fprintf('%.3g ',Lambda);
            fprintf('\nKernel Par = '); fprintf('%.3g ',KP);
            fprintf('\nDecomp. Size = '); fprintf('%d ',RP);
            % For each regularization param
            for li = 1:numel(Lambda),
                lambda = Lambda(li);
                fprintf('\nLambda = %.3g',lambda);
                % For each kernel param
                for kpi = 1:numel(KP),
                    kp = KP(kpi);
                    kernel_fun = @(X) rbf(X',kp);
                    fprintf('\n\tKernel Par = %.3g',kp);
                    % Run nlmc on each data subset
                    for nD = 1:nTrials,
                        fprintf('\n\t\t Trial %d/%d --> R = ',nD,nTrials);
                        % Ntrain & Ntest
                        nTrain = round(numel(data.splits{nD})*pp/100);
                        nTest = numel(data.splits{nD})-nTrain;
                        % Get features
                        splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                        % Get labels
                        splitTrainLabels = data.labels{nD}(1:nTrain);
                        splitTestLabels = data.labels{nD}(nTrain+1:end);
                        % Kernelize
                        KX = kernel_fun(splitFeatures');
                        for rpi = 1:numel(RP),
                            rp = RP(rpi);
                            % Learn nlmc
                            Qopt = nlmc_learn(KX,splitTrainLabels,rp,lambda);
                            % Infer labels with nlmc
                            [e_test_labels, e_train_labels] = nlmc_infer(Qopt,splitTrainLabels,rp,lambda);
                            % Binarize
                            e_test_labels = 2*(e_test_labels > 0)-1;
                            e_train_labels = 2*(e_train_labels > 0)-1;
                            % Compute accuracy on training and testing
                            e_aux = e_train_labels;
                            t_aux = splitTrainLabels;
                            accuracyTrainEmotion(nD,li,kpi,rpi) = sum( e_aux-t_aux == 0 )/nTrain;
                            e_aux = e_test_labels;
                            t_aux = splitTestLabels;
                            accuracyTestEmotion(nD,li,kpi,rpi) = sum( e_aux-t_aux == 0 )/nTest;
                            % Output
                            fprintf('%d, ',rp);
                        end
                    end
                end
            end
            % Save
            save([resultsPath  resultsName],'KP','Lambda','RP','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        case 'lmc'
	    % Path
	    addpath(nlmcPath);
            % Parameters
            RP = DecompositionSize;
            Lambda = RegularizationMC;
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(Lambda),numel(RP));
            accuracyTestEmotion = zeros(nTrials,numel(Lambda),numel(RP));
            % Output
            fprintf('nlmc-LIN === Starting cross-validation with:');
            fprintf('\nLambda = '); fprintf('%.3g ',Lambda);
            fprintf('\nDecomp. Size = '); fprintf('%d ',RP);
            % For each regularization param
            for li = 1:numel(Lambda),
                lambda = Lambda(li);
                fprintf('\nLambda = %.3g',lambda);
                % Kernel
                kernel_fun = @(X) X'*X;
                % Run nlmc on each data subset
                for nD = 1:nTrials,
                    fprintf('\n\t Trial %d/%d --> R = ',nD,nTrials);
                    % Ntrain & Ntest
                    nTrain = round(numel(data.splits{nD})*pp/100);
                    nTest = numel(data.splits{nD})-nTrain;
                    % Get features
                    splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                    % Get labels
                    splitTrainLabels = data.labels{nD}(1:nTrain);
                    splitTestLabels = data.labels{nD}(nTrain+1:end);
                    % Kernelize
                    KX = kernel_fun(splitFeatures');
                    for rpi = 1:numel(RP),
                        rp = RP(rpi);
                        % Learn nlmc
                        Qopt = nlmc_learn(KX,splitTrainLabels,rp,lambda);
                        % Infer labels with nlmc
                        [e_test_labels, e_train_labels] = nlmc_infer(Qopt,splitTrainLabels,rp,lambda);
                        % Binarize
                        e_test_labels = 2*(e_test_labels > 0)-1;
                        e_train_labels = 2*(e_train_labels > 0)-1;
                        % Compute accuracy on training and testing
                        e_aux = e_train_labels;
                        t_aux = splitTrainLabels;
                        accuracyTrainEmotion(nD,li,rpi) = sum( e_aux-t_aux == 0 )/nTrain;
                        e_aux = e_test_labels;
                        t_aux = splitTestLabels;
                        accuracyTestEmotion(nD,li,rpi) = sum( e_aux-t_aux == 0 )/nTest;
                        % Output
                        fprintf('%d, ',rp);
                    end
                end
            end
            % Save
            save([resultsPath  resultsName],'Lambda','RP','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        case 'tsvm'
            % Change dir
            currentPath = pwd;
            cd(tsvmPath);
            % Regularization parameter
            CP = RegularizationTSVM;
            KP = RBFKernelParam;
            fprintf('TSVM === Starting cross-validation with:');
            fprintf('\nRegularization = '); fprintf('%.3g ',CP);
            fprintf('\nKernel Par = '); fprintf('%.3g ',KP);
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(CP),numel(KP));
            accuracyTestEmotion = zeros(nTrials,numel(CP),numel(KP));
            % Loop
            for cpi = 1:numel(CP),
                cp = CP(cpi);
                fprintf('\nRegParam = %.3g',cp);
                % leave-one-out
                for kpi = 1:numel(KP),
                    kp = KP(kpi);
                    fprintf('\n\tKernel Par = %.3g',kp);
                    for nD = 1:nTrials,
                        fprintf('\n\t Trial %d/%d',nD,nTrials);
                        % Ntrain & Ntest
                        nTrain = round(numel(data.splits{nD})*pp/100);
                        nTest = numel(data.splits{nD})-nTrain;
                        % Get features
                        splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                        % Get labels
                        splitTrainLabels = data.labels{nD}(1:nTrain);
                        splitTestLabels = zeros(size(data.labels{nD}(nTrain+1:end)));
                        % Train
                        Xall=splitFeatures;
                        Yall=[splitTrainLabels'; splitTestLabels'];
                        svmlwrite('training.dat', Xall, Yall);
                        systemCommand = ['./svm_learn -c ' num2str(cp) ' -t 2 -g ' num2str(1/kp.^2) ' training.dat model'];
                        [~,~] = system(systemCommand);
%                         fprintf(' %s',systemCommand);
                        % Classify training
                        Xtrain = splitFeatures(1:nTrain,:);
                        Ytrain = splitTrainLabels';
                        svmlwrite('classifyTrain.dat',Xtrain,Ytrain);
                        [~,~] = system('./svm_classify classifyTrain.dat model predictionsTrain.dat');
                        % Classify test
                        Xtest = splitFeatures(nTrain+1:end,:);
                        Ytest = splitTestLabels';
                        svmlwrite('classifyTest.dat',Xtest,Ytest);
                        [~,~] = system('./svm_classify classifyTest.dat model predictionsTest.dat');
                        % Load results and performance measure
                        aux = load('predictionsTrain.dat');
                        e_train = sign(aux);
                        aux = load('predictionsTest.dat');
                        e_test = sign(aux);
                        accuracyTestEmotion(nD,cpi,kpi) = sum( Ytest - e_test == 0)/nTest;
                        accuracyTrainEmotion(nD,cpi,kpi) = sum( Ytrain - e_train == 0)/nTrain;
                    end
                end
            end
            % Change dir back
            cd(currentPath);
            % Save
            save([resultsPath  resultsName],'CP','KP','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        case 'svm'
            % Path
            addpath(svmPath);
            % Regularization parameter
            CP = RegularizationSVM;
            KP = RBFKernelParam;
            fprintf('SVM === Starting cross-validation with:');
            fprintf('\nRegularization = '); fprintf('%.3g ',CP);
            fprintf('\nKernel Par = '); fprintf('%.3g ',KP);
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(CP),numel(KP));
            accuracyTestEmotion = zeros(nTrials,numel(CP),numel(KP));
            % Loop
            for cpi = 1:numel(CP),
                cp = CP(cpi);
                fprintf('\nRegParam = %.3g',cp);
                % leave-one-out
                for kpi = 1:numel(KP),
                    kp = KP(kpi);
                    fprintf('\n\tKernel Par = %.3g',kp);
                    for nD = 1:nTrials,
                        fprintf('\n\t Trial %d/%d',nD,nTrials);
                        % Ntrain & Ntest
                        nTrain = round(numel(data.splits{nD})*pp/100);
                        nTest = numel(data.splits{nD})-nTrain;
                        % Get features
                        splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                        Xtrain = splitFeatures(1:nTrain,:);
                        Xtest = splitFeatures(nTrain+1:end,:);                        
                        % Get labels
                        Ytrain = data.labels{nD}(1:nTrain)/2+1.5;
                        Ytest = data.labels{nD}(nTrain+1:end)/2+1.5;
                        % Train
                        model = svmtrain(Ytrain',Xtrain,['-c ' num2str(cp) ' -t 2 -g ' num2str(1/kp.^2) ' -q']);
                        % Evaluate classifier
                        e_train = svmpredict(Ytrain',Xtrain,model,'-q');
                        e_test = svmpredict(Ytest',Xtest,model,'-q');
                        % Save results
                        accuracyTestEmotion(nD,cpi,kpi) = sum( Ytest' - e_test == 0)/nTest;
                        accuracyTrainEmotion(nD,cpi,kpi) = sum( Ytrain' - e_train == 0)/nTest;
                    end
                end
            end
            % Save
            save([resultsPath  resultsName],'CP','KP','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        case 'lasso'
            % Path
            addpath(genpath(slepPath));
            % Regularization parameter
            CP = RegularizationLASSO;
            fprintf('LASSO === Starting cross-validation with:');
            fprintf('\nRegularization = '); fprintf('%.3g ',CP);
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(CP));
            accuracyTestEmotion = zeros(nTrials,numel(CP));
            % Loop
            for cpi = 1:numel(CP),
                cp = CP(cpi);
                fprintf('\nRegParam = %.3g',cp);
                % leave-one-out
                for nD = 1:nTrials,
                    fprintf('\n\t Trial %d/%d',nD,nTrials);
                    % Ntrain & Ntest
                    nTrain = round(numel(data.splits{nD})*pp/100);
                    nTest = numel(data.splits{nD})-nTrain;
                    % Get features
                    splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                    Xtrain = splitFeatures(1:nTrain,:);
                    Xtest = splitFeatures(nTrain+1:end,:); 
                    % Get labels
                    Ytrain = data.labels{nD}(1:nTrain);
                    Ytest = data.labels{nD}(nTrain+1:end);
                    % Train
                    classifier =  LeastC(Xtrain,Ytrain',cp);
                    % Evaluate classifier
                    e_train = sign(Xtrain*classifier);
                    e_test = sign(Xtest*classifier);
                    % Save results
                    accuracyTestEmotion(nD,cpi) = sum( Ytest' - e_test == 0)/nTest;
                    accuracyTrainEmotion(nD,cpi) = sum( Ytrain' - e_train == 0)/nTrain;
                end
            end
            % Save
            save([resultsPath  resultsName],'CP','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        case 'group-lasso'
            % Path
            addpath(genpath(slepPath));
            % Regularization parameter
            L1 = RegularizationGroupLASSO;
            L2 = L1;
            fprintf('GROUP-LASSO === Starting cross-validation with:');
            fprintf('\nLambda1 = '); fprintf('%.3g ',L1);
            fprintf('\nLambda2 = '); fprintf('%.3g ',L2);
            % Results
            accuracyTrainEmotion = zeros(nTrials,numel(L1),numel(L2));
            accuracyTestEmotion = zeros(nTrials,numel(L1),numel(L2));
            % Loop
            for l1i = 1:numel(L1),
                l1 = L1(l1i);
                fprintf('\n\tL1 = %.3g',l1);
                for l2i = 1:numel(L2),
                    l2 = L2(l2i);
                    fprintf('\n\t\tL2 = %.3g',l2);
                    % leave-one-out
                    for nD = 1:nTrials,
                        fprintf('\n\t Trial %d/%d',nD,nTrials);
                        % Ntrain & Ntest
                        nTrain = round(numel(data.splits{nD})*pp/100);
                        nTest = numel(data.splits{nD})-nTrain;
                        % Get features
                        splitFeatures = data.allPCAFeatures(data.splits{nD},:);
                        Xtrain = splitFeatures(1:nTrain,:);
                        Xtest = splitFeatures(nTrain+1:end,:); 
                        % Get labels
                        Ytrain = data.labels{nD}(1:nTrain);
                        Ytest = data.labels{nD}(nTrain+1:end);
                        opts.rFlag=1;
                        classifier =  mc_sgLeastR(Xtrain,Ytrain',[l1,l2],opts);
                        % Evaluate classifier
                        e_train = sign(Xtrain*classifier);
                        e_test = sign(Xtest*classifier);
                        % Save results
                        accuracyTestEmotion(nD,l1i,l2i) = sum( Ytest' - e_test == 0)/nTest;
                        accuracyTrainEmotion(nD,l1i,l2i) = sum( Ytrain' - e_train == 0)/nTrain;
                    end
                end
            end
            % Save
            save([resultsPath  resultsName],'L1','L2','accuracyTrainEmotion','accuracyTestEmotion');
            fprintf('\nResults saved\n');
        otherwise
            fprintf('Method %s not ready.\n',method);
    end
end
