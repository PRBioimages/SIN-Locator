function idx = fs(trdata, target_i, fsMethod, classifyPath, fType, fType_i, k, i, indLabel)
% feature selection
% trdata  n*d matrix
% target_i  n*1 matrix (0 1)

target_i(target_i==0) = -1;
u = sort(unique(target_i),'descend');

switch fsMethod 
    case 'nofs'
        idx = 1:size(trdata,2);
        
    case 'sda'
        if strcmp(fType, fType_i)  % if only one descriptor
            idxsdaPath = [classifyPath '/sdalog/' fType_i '_fold' num2str(i) '_label' num2str(indLabel) '.mat'];
            logfilename = [classifyPath '/sdalog/' fType_i '_fold' num2str(i) '_label' num2str(indLabel) '_sdalog.txt'];
        elseif length(strfind(fType, '+'))==1 % if two descriptors
            idxsdaPath = [classifyPath '/sdalog/' fType '_' fType_i '_fold' num2str(i) '_label' num2str(indLabel) '.mat'];
            logfilename = [classifyPath '/sdalog/' fType '_' fType_i '_fold' num2str(i) '_label' num2str(indLabel) '_sdalog.txt'];
        elseif length(strfind(fType, '+'))==2 % if three descriptors
            idxsdaPath = [classifyPath '/sdalog/' fType '_feat' num2str(k) '_fold' num2str(i) '_label' num2str(indLabel) '.mat'];
            logfilename = [classifyPath '/sdalog/' fType '_feat' num2str(k) '_fold' num2str(i) '_label' num2str(indLabel) '_sdalog.txt'];
        end
        if exist(idxsdaPath, 'file')
            load(idxsdaPath);
        else
            feat = cell(1,length(u));
            for j = 1:length(u) % for each class
                feat{j} = trdata(target_i==u(j),:);
            end
            
            idx = ml_stepdisc( feat,logfilename); % selected features
            idx = unique(idx);
            save(idxsdaPath, 'idx');
        end
end