function [trdata_i, target_i] = balanceData(trdata_i, target_i, u, baMethod)
% data rebalance

if strcmp(baMethod, 'noba')
    return;
end

usamples = zeros(1,length(u));
uInd = cell(1,length(u));
for i = 1:length(u)
    uInd{i} = find(target_i==u(i));
    usamples(1,i) = length(find(target_i==u(i)));
end
uratio = max(usamples)/min(usamples);

if strcmp(baMethod, 'smote')
    % up sampleing using SMOTE
    while  uratio>4
        % use SMOTE
        dataSmote = mySMOTE([trdata_i target_i], 5, sort(target_i));
        trdata_i = dataSmote(:,1:end-1);
        target_i = dataSmote(:,end);
        for i = 1:length(u)
            usamples(1,i) = length(find(target_i==u(i)));
        end
        uratio = max(usamples)/min(usamples);
    end
end

if strcmp(baMethod, 'down')
    % down sampling 
    uIndDown = cell(1,length(u));
    while uratio>4
        trdata = [];  target = [];
        num = floor(mean(usamples));
        for i = 1:length(u)
            if length(uInd{i})<num
                uIndDown{i} = uInd{i};
            else
                uIndDown{i} = uInd{i}(randperm(length(uInd{i}),num),1);
            end
            trdata =  [trdata; trdata_i(uIndDown{i},:)];
            target = [target; target_i(uIndDown{i},:)];
        end
        trdata_i = trdata;
        target_i = target;

        % decide whether to down sample again
        for i = 1:length(u)
            uInd{i} = find(target_i==u(i));
            usamples(1,i) = length(find(target_i==u(i)));
        end
        uratio = max(usamples)/min(usamples);
    end
end
