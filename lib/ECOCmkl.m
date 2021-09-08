function [outputs, bestparas] = ECOCmkl(trdata, tedata, trlabel, idx, indcom, singleInd, fType, strs, para_i,telabel)

% reconstruct idx
if size(idx,2)~=length(singleInd)
    idfeat = cell(size(idx,1), size(indcom,1)+length(singleInd));
        for i = 1:size(idfeat,1)
            for j = 1:size(indcom,1)
                for k = 1:length(indcom(j,:))
                    idfeat{i,j}{1,k} = idx{i, indcom(j,k)};
                end
            end
            for l = 1:length(singleInd)
                idfeat{i,j+l} = idx{i,singleInd(l)};
            end
        end      
else
    idfeat = idx;
end

% gamma for different features
% gammasearch = 2.^[-4:5]; 
% Csearch = 2.^[-5:5];
gammasearch = 2.^[-3:2]; 
Csearch = 2.^[-2:3];

% search gamma
gammaSearchNum = min([1000, ceil(0.8*length(gammasearch)^length(singleInd))]); % repeat time
para = zeros(gammaSearchNum, size(idx,2)+1);   % one gamma for each feature set
outputs_i = cell(gammaSearchNum, length(Csearch));
accs = zeros(gammaSearchNum, length(Csearch));
bestbeta = cell(gammaSearchNum, length(Csearch));

% only use training set to search gamma and C
ind = 1:2:size(trdata,1);
trdata_1 = trdata(ind, :);
trdata_2 = trdata(setdiff(1:size(trdata,1), ind), :);
trlabel_1 = trlabel(:, ind);
trlabel_2 = trlabel(:, setdiff(1:size(trdata,1), ind));

for i = 1:gammaSearchNum   % search gamma
   % randomly gamma combination
   flag = 0;
   while sum(para(i,:))==0 || flag==1
       para_j = [];
        for j = 1:size(indcom,1)
            para_j = [para_j para_i(j, 1:length(indcom(j,:)))];
        end
        for j = 1:length(singleInd)
            para_j = [para_j gammasearch(randi(length(gammasearch)))];
        end
        para(i,1:end-1) = para_j;
        flag = 0;
        for j = 1:i-1 % if same as previous parameter combinations
            if sum(para(i,:)~=para(j,:))==0 % same
                flag = 1;
                break;
            end
        end
   end
   
   for j = 1:length(Csearch)
        tic
        para(i,end) = Csearch(1,j);

        % classification
        outputs_i{i,j} = zeros(size(trlabel_2));
        betas = [];
        betasknown = para_i(:, (size(indcom,2)+2):end);
                
        for k = 1:size(trlabel_2,1)  
            base_params.idx=idx(k,:);
            base_params.idfeat=idfeat(k,:);
            base_params.gamma=para(i,1:end-1);
            base_params.C = para(i,end);
            base_params.indcom = indcom;
            base_params.betasknown = betasknown(:, (size(indcom,2)*(k-1)+1):(size(indcom,2)*k));
            
            % parameters 
            Parameters.coding='OneVsOne';
            Parameters.base='svm_train_multi_kernel';
            Parameters.base_test='svm_test_multi_kernel';
            Parameters.base_test_params= base_params;
            Parameters.base_params=base_params;
            % classification
            trlabel_j = trlabel_1(k,:);    
            [Classifiers,Parameters] = ECOCTrain(trdata_1,trlabel_j,Parameters); 
            [X,Labels,temp_Values,confusion] = ECOCTest(trdata_2,Classifiers,Parameters);
            if Classifiers{1,1}.classifier.result.model.Label(1,1) == 1
                outputs_i{i,j}(k,:) = -X';
            else
                outputs_i{i,j}(k,:) = X';
            end
            betas = [betas Classifiers{1, 1}.classifier.result.beta];
        end
        % evaluate
        predictedLabel = getLabelset(outputs_i{i,j}, 2, 5, 0, 0, trlabel_2);
        evalCriteria = evalModel(outputs_i{i,j}, predictedLabel, trlabel_2, 1);
        accs(i,j) = evalCriteria.accuracy;
        bestbeta{i,j} = betas;
        
        toc
   end
end
[a, row] = max(accs,[],1);
[bestacc, col] = max(a);
row = row(col);
para_ij = para(row,:);
para_ij(1,end) = Csearch(1,col);
bestparas = [para_ij bestbeta{row,col}];

% use the searched parameter to train model
outputs = zeros(size(telabel));
for k = 1:size(trlabel,1)  
    base_params.idx=idx(k,:);
    base_params.idfeat=idfeat(k,:);
    base_params.gamma=para_ij(1,1:end-1);
    base_params.C = para_ij(1,end);
    base_params.indcom = indcom;

    % parameters 
    Parameters.coding='OneVsOne';
    Parameters.base='svm_train_multi_kernel';
    Parameters.base_test='svm_test_multi_kernel';
    Parameters.base_test_params= base_params;
    Parameters.base_params=base_params;
    
    % classification
    trlabel_j = trlabel(k,:);    
    [Classifiers,Parameters] = ECOCTrain(trdata,trlabel_j,Parameters); 
    [X,Labels,temp_Values,confusion] = ECOCTest(tedata,Classifiers,Parameters);
    if Classifiers{1,1}.classifier.result.model.Label(1,1) == 1
        outputs(k,:) = -X';
    else
        outputs(k,:) = X';
    end
end
