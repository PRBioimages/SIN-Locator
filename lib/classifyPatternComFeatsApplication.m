function classifyPatternComFeatsApplication(fType, cType, fsAllMethod, baMethod, oriPath, featPathApplication, classifyAllPath, multiFlag)
% feed different features into classifiers, and save classification results

classifyPath = [classifyAllPath '/multiLabelResults']; % 8_classification/multiLabelResults

if ~exist(classifyPath, 'dir')
    mkdir(classifyPath);
end

writePath = [classifyPath '/' fType '_' fsAllMethod cType '_' baMethod '.mat']; % result path
if exist(writePath,'file')
    return
end    
    
tmpfile = ['./tmp/' fType '_' fsAllMethod cType '_' baMethod '.txt']; % tmp txt
if exist(tmpfile, 'file')
    return;
end
fid = fopen(tmpfile, 'w');

indi = strfind(fType, fType(1:2)); % 'IF' or 'IH'
indadd = [];
indcom = [];
for i = 2:length(indi)
    if ~strcmp(fType(indi(i)-1), '+')
        indadd = [indadd indi(i)];  % position adding +
        indcom = [indcom; [i-1 i]]; % combination index
    end
end
singleInd = setdiff(1:length(indi), reshape(indcom, 1, [])); % feat set index (no combination)

strs = cell(length(indi),1);
ind = strfind(fType, '+'); 
if length(ind)==1  && contains(fType, 'perPro') && ~contains(fType, 'seq')% only images
    % load training data
    load(['./data/7_constructedData/' fType(1:ind-1) '.mat'],'feats','protInd'); %  image feature set 1
    strs{1,1} = fType(1:ind-1);
    protInd1 = protInd;
    feats1 = feats;
    load(['./data/7_constructedData/' fType(ind+1:end) '.mat'],'feats','labels','protInd'); %  image feature set 2
    strs{2,1} = fType(ind+1:end);
    [protIndsect, ia, ib] = intersect(protInd1, protInd);
    trdata = [feats1(ia,:) feats(ib,:)];
    trlabel = labels(:,ib);
    idx = getIdx(fType, strs, size(labels,1), size(feats1,2));  % sda index
    % load test data (Application data)
    load([featPathApplication '/' fType(1:ind-1) '.mat']); % image feature set 1
    protInd1 = protInd;
    feats1 = feats;
    load([featPathApplication '/' fType(ind+1:end) '.mat']); % image feature set 2
    [protIndsect, ia, ib] = intersect(protInd1, protInd);
    tedata = [feats1(ia,:)  feats(ib,:)];
    telabel = labels(:,ib);
    testProtInd = protInd(ib,:);
    % classification
    [trdata, tedata] = featnorm(trdata, tedata);
    [predictedScore, parameters] = ECOCmkl(trdata, tedata, trlabel, idx, indcom, singleInd, fType, strs, [], telabel);
     
elseif length(ind)==2  % image+seq+ppi
    fTypeP = ['+' fType '+']; % feat types
    indadd = indadd+(0:length(indadd)-1)+1;
    for i = 1:length(indadd)
        fTypeP = [fTypeP(1:indadd(i)-1) '+' fTypeP(indadd(i):end)];
    end
    indp = strfind(fTypeP, '+');
    for i = 1:length(indp)-1 % for each type of features
        fTypeP_i  = fTypeP(indp(i)+1:indp(i+1)-1);
        strs{i,1} = fTypeP_i;
    end
    
    % load training data
    featsAll = cell(length(strs),1);
    protIndAll = cell(length(strs),1);
    protIndsect =1:10000;
    dimFeat = [];
    for i = 1:length(strs)
        if strcmp(strs{i,1}(end-3:end), '_seq')
            load(['./data/7_constructedData/' strs{i,1} 'uence_feats.mat']);
        else
            load(['./data/7_constructedData/' strs{i,1} '.mat']);
        end
        featsAll{i,1} = feats;  dimFeat = [dimFeat size(feats,2)];
        protIndAll{i,1} = protInd;
        protIndsect = intersect(protIndsect, protInd);
    end
    trdata = [];
    for i = 1:length(protIndAll)
        [~, ia, ~] = intersect(protIndAll{i,1}, protIndsect);
        trdata = [trdata featsAll{i,1}(ia,:)];
    end
    load(['./data/7_constructedData/' strs{1,1} 'uence_feats.mat'], 'labels', 'protInd');
    [~, ia, ~] = intersect(protInd, protIndsect);
    trlabel = labels(:,ia);
    numLabel = size(labels,1);
    idx = getIdx(fType, strs, numLabel, dimFeat); % sda idx
    % load testing data
    featsAll = cell(length(strs),1);
    protIndAll = cell(length(strs),1);
    protIndsect =1:10000;
    for i = 1:length(strs)
        if strcmp(strs{i,1}(end-3:end), '_seq')
            load([featPathApplication '/' strs{i,1} 'uence_feats.mat']);
        else
            load([featPathApplication '/' strs{i,1} '.mat']);
        end
        featsAll{i,1} = feats;
        protIndAll{i,1} = protInd;
        protIndsect = intersect(protIndsect, protInd);
    end
    tedata = [];
    for i = 1:length(protIndAll)
        [~, ia, ~] = intersect(protIndAll{i,1}, protIndsect);
        tedata = [tedata featsAll{i,1}(ia,:)];
    end
    load([featPathApplication '/' strs{1,1} 'uence_feats.mat'], 'labels', 'protInd');
    [~, ia, ~] = intersect(protInd, protIndsect);
    telabel = labels(:,ia);
    testProtInd = protInd(ia,:);
    
    % classification
    [trdata, tedata] = featnorm(trdata, tedata);
    para = [];
    if ~isempty(indcom)
        for i = 1:size(indcom,1)
            lopath = [ oriPath '/multiLabelResults/' strs{indcom(i,1)}];
            for j = 2:length(indcom(i,:))
                lopath = [lopath  '+' strs{indcom(i,j)}];
            end
            lopath = [lopath '_' cType '_' baMethod '.mat'];
            load(lopath, 'parameters');
            para = [para; parameters];  % parameters searched
        end     
    end
    para_i = para(5*[0:1:size(indcom,1)-1]+i,:);
    [predictedScore, parameters] = ECOCmkl(trdata, tedata, trlabel, idx, indcom, singleInd, fType, strs, para_i, telabel);
end

predictedLabel = getLabelset(predictedScore, 2, 2, 0.74, 0.94, telabel);
evalCriteria = evalModel(predictedScore, predictedLabel, telabel, 1);
acc = evalCriteria.accuracy;
f1s = mean(evalCriteria.F1_score);
if strcmp(cType, 'ECOCmkl')
    save(writePath, 'evalCriteria' , 'acc', 'f1s', 'predictedScore', 'predictedLabel', 'parameters', 'telabel', 'testProtInd');
else
   save(writePath, 'evalCriteria' , 'acc', 'f1s', 'predictedScore', 'predictedLabel', 'telabel');
end

fclose(fid);
delete(tmpfile);


function idxAll = getIdx(fType, strs, numLabel, dimFeat)
% load sda ind
sdafold = 5;
idxAll = cell(numLabel,length(strs));
if length(strs)==2
    for i = 1:numLabel
        for j = 1:length(strs)
            idxAll{i,j} = [];
            for k = 1:sdafold
                p = ['./data/8_classification/sdalog/' fType '_' strs{j,1} '_fold' num2str(k) '_label' num2str(i) '.mat'];
                load(p,'idx');
                if j==2
                    idx = idx+dimFeat;
                end
                idxAll{i,j} = [idxAll{i,j} idx];
            end
            idxAll{i,j} = unique(idxAll{i,j});
        end
    end           
elseif length(strs)==5
    for i = 1:numLabel
        for j = 1:length(strs)
            idxAll{i,j} = [];
            for k = 1:sdafold
                p = ['./data/8_classification/sdalog/' fType '_feat' num2str(j) '_fold' num2str(k) '_label' num2str(i) '.mat'];
                if exist(p,'file')
                    load(p,'idx');
                else
                    idx = 1:dimFeat(j);
                end
                if j>=2 % dim+
                    idx = idx+sum(dimFeat(1:j-1));
                end
                idxAll{i,j} = [idxAll{i,j} idx];
            end
            idxAll{i,j} = unique(idxAll{i,j});
        end
    end    
end

