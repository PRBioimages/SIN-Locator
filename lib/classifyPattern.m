function classifyPattern(fType, cType, baMethod, featPath, classifyAllPath)
% feed different features into classifiers, and save classification results

classifyPath = [classifyAllPath '/multiLabelResults']; % 8_classification/multiLabelResults

if ~exist(classifyPath, 'dir')
    mkdir(classifyPath);
end

fTypeP = ['+' fType '+']; % feat types
ind = strfind(fTypeP, '+');
if length(ind)>2 % whether multiple features 
    mklrun = 1;
else
    mklrun = 0;
    cType = 'brsvm'; % if use single-type features
end 

writePath = [classifyPath '/' fType '_' cType '_' baMethod '.mat']; % result path
if exist(writePath,'file')
    return
end    
    
tmpfile = ['./tmp/' fType '_' cType '_' baMethod '.txt']; % tmp txt
if exist(tmpfile, 'file')
    return;
end
fid = fopen(tmpfile, 'w');

ind = strfind(fTypeP, fTypeP(2:3)); % 'IF' or 'IH'
indadd = [];
indcom = [];
for i = 2:length(ind)
    if ~strcmp(fTypeP(ind(i)-1), '+')
        indadd = [indadd ind(i)];  % position adding +
        indcom = [indcom; [i-1 i]]; % combination index
    end
end
singleInd = setdiff(1:length(ind), reshape(indcom, 1, [])); % feat set index (no combination)

indadd = indadd+(0:length(indadd)-1);
for i = 1:length(indadd)
    fTypeP = [fTypeP(1:indadd(i)-1) '+' fTypeP(indadd(i):end)];
end
      
ind = strfind(fTypeP, '+');

% load features 
featsAll = cell(length(ind)-1,1);  
featDimAll = zeros(length(ind)-1,1);
protIndAll = cell(length(ind)-1,1);  
labelsIndAll = cell(length(ind)-1,1);  
labelsAll = cell(length(ind)-1,1);  
fsMethods = cell(length(ind)-1,1);  
predictedScoreAll = cell(length(ind)-1,1);  % only used when 'vote'
strs = cell(length(ind)-1, 1);
for i = 1:length(ind)-1 % for each type of features
    fTypeP_i  = fTypeP(ind(i)+1:ind(i+1)-1);
    strs{i,1} = fTypeP_i;
    switch fTypeP_i
        case 'IF_seq' % only use sequence features (139D) for IF dataset
            load([featPath '/IF_sequence_feats.mat']);   
            fsMethods{i,1} = 'nofs';  % no feature selection
        case 'IHC_seq' % only use sequence features (139D) for IHC dataset
            load([featPath '/IHC_sequence_feats.mat']);
            fsMethods{i,1} = 'nofs'; 
        otherwise
            load([featPath '/' fTypeP_i '.mat']); % images
            fsMethods{i,1} = 'sda'; 
    end
    if ~isempty(strfind(fTypeP_i, 'IHC_perPro_aveFeats')) ...
        fsMethods{i,1} = 'nofs'; 
    end

    featsAll{i,1} = feats;
    featDimAll(i,1) = size(feats,2);
    protIndAll{i,1} = protInd;
    labelsIndAll{i,1} = labelsInd;
    labelsAll{i,1} = labels;

    % if cType='vote', combined in decision level
    if strcmp(cType, 'vote') && length(ind)>2
        pathload = [classifyPath '/' fTypeP_i '_brsvm_' baMethod '.mat'];
        if exist(pathload, 'file')
            load(pathload,'predictedScore');
            predictedScoreAll{i,1} = predictedScore;
        else
            error('Please run BR svm models first! ');
        end
    end
end

feats = featsAll{1,1}; % first type of feats
protInd = protIndAll{1,1};
labelsInd = labelsIndAll{1,1};
labels = labelsAll{1,1};
predictedScore = predictedScoreAll{1,1};
for i = 2:length(featsAll) % combine different types of features
    % sample index
    [protInd, ia, ib] = intersect(protIndAll{i,1}, protInd);
    if isempty(featsAll{i,1}) % if is PPI features, it is empty
        feats_i = [];
    else
        feats_i = featsAll{i,1}(ia,:);
    end
    feats = [feats(ib, :), feats_i];
    labelsInd = labelsInd(1,ib);
    labels = labels(:, ib);

    % if cType='vote', combined in decision level
    if strcmp(cType, 'vote') && length(ind)>2
        predictedScore = predictedScore(:,ib)+predictedScoreAll{i}(:,ia);
    end
end

% if cType='vote', combined in decision level
if strcmp(cType, 'vote') && length(ind)>2
    predictedScore = predictedScore/length(predictedScoreAll);
    [predictedLabel, cons, sita] = getLabelset(predictedScore, 2, 2, 0.74, 0.94, labels);
    evalCriteria = evalModel(predictedScore, predictedLabel, labels, 1);
    acc = evalCriteria.accuracy;
    f1s = mean(evalCriteria.F1_score);
    save(writePath, 'evalCriteria' , 'acc', 'f1s', 'predictedScore', 'predictedLabel');
    fclose(fid);
    delete(tmpfile);

    return
end

% classification
foldNum = 5;  % 5-fold cross validation
sampleNum = size(protInd,1);
tmp = rem(1:sampleNum,foldNum);
tmp = tmp(randperm(sampleNum)); % random split

if strcmp(cType, 'ECOCmkl')
    para = [];
    if ~isempty(indcom)
        for i = 1:size(indcom,1)
            lopath = [ classifyPath '/' strs{indcom(i,1)}];
            for j = 2:length(indcom(i,:))
                lopath = [lopath  '+' strs{indcom(i,j)}];
            end
            lopath = [lopath '_' cType '_' baMethod '.mat'];
            load(lopath, 'parameters');
            para = [para; parameters];  % parameters searched
        end
    end
end

parameters = [];
f1s = zeros(foldNum,1);
predictedScore = zeros(size(labels));
predictedLabel = zeros(size(labelsInd));
for i = 1:foldNum
    fprintf(['Running ' cType ' fold ' num2str(i) '...\n']);
    % 1. split training and testing set
    teInd = find(tmp==i-1);
    trInd = setdiff(1:sampleNum, teInd);
    if ~isempty(feats)
        trdata = feats(trInd,:);  tedata = feats(teInd,:);  
    else
        trdata = [];  tedata = [];  
    end    
    % multi label
    trlabel = labels(:, trInd); % train
    telabel = labels(:, teInd); % test
    
    % 2. normalization and feature selection
    [trdata, tedata] = featnorm(trdata, tedata);  % normalization
    % feature selection 
    for  j = 1:size(labels,1) % for each class
        for k = 1:length(fsMethods) % for each feature set
            indfeat_k = (sum(featDimAll(1:k-1,1))+1):sum(featDimAll(1:k,1));
            idx{j,k} = fs(trdata(:,indfeat_k), trlabel(j,:), fsMethods{k,1}, classifyAllPath, fType, strs{k,1}, k, num2str(i), j); 
            idx{j,k} = idx{j,k} + sum(featDimAll(1:k-1,1));
        end
    end
    
    % 3. classification       
    % multi label classification
    switch cType 
        case 'brsvm'
            % use BR method
            outputs = trainBRSVMmodel(trdata, tedata, trlabel, idx, baMethod);
        case 'ECOCmkl'
            % ECOC with multi-kernel learning
            if isempty(indcom)
                para_i = [];
            else
                para_i = para(foldNum*[0:1:size(indcom,1)-1]+i,:);
            end
            [outputs, bestpara] = ECOCmkl(trdata, tedata, trlabel, idx, indcom, singleInd, fType, strs, para_i, telabel);
            parameters = [parameters; bestpara];
        case 'dnn'
            outputs = featdnn(trdata, tedata, trlabel, idx, cType);
    end  
    predictedScore(:, teInd) = outputs;      
end

[predictedLabel, cons, sita] = getLabelset(predictedScore, 2, 2, 0.74, 0.94, labels);
evalCriteria = evalModel(predictedScore, predictedLabel, labels,1);
acc = evalCriteria.accuracy;
f1s = mean(evalCriteria.F1_score);

if strcmp(cType, 'ECOCmkl') 
    save(writePath, 'evalCriteria' , 'acc', 'f1s', 'predictedScore', 'predictedLabel', 'parameters');
else
    save(writePath, 'evalCriteria' , 'acc', 'f1s', 'predictedScore', 'predictedLabel');
end

fclose(fid);
delete(tmpfile);

