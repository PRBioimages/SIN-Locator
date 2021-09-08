function prelabels = trainBRSVMmodel(trdata, tedata, trlabel, idx, baMethod)
% BR method using SVM model

% selected features
if size(idx,1)~=size(trlabel,1)
    error('Wrong idx size!\n');
end
if size(idx,2)>1
    for i = 1:size(idx,1)
        for j = 2:size(idx,2)
            idx{i,1} = [idx{i,1} idx{i,j}];
        end
    end
end
    
% classification
prelabels = zeros(size(trlabel,1), size(tedata,1));
for i=1:size(trlabel,1) % for each class
    target_i = trlabel(i,:)'; % labels of class i
    
    % feature selection
    trdata_i = trdata(:, idx{i,1});
    tedata_i = tedata(:, idx{i,1});
    
    % train SVM model
     score = trainSVMmodel(trdata_i, tedata_i, target_i, baMethod);
     prelabels(i,:) = score(:,2)';
end