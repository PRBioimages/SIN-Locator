function [trdata, tedata] = featnorm(trdata, tedata)
MItrain = min(trdata,[],1);
MAtrain = MItrain;

for i=1:size(trdata,2) % training data
        trdata(:,i) = trdata(:,i) - MItrain(i);
        MAtrain(i) = max(trdata(:,i));
        if MAtrain(i)~=0
            trdata(:,i) = trdata(:,i) / MAtrain(i);
        else
            trdata(:,i) = 0.5.*ones(size(trdata,1),1);
        end
end
trdata = double( trdata*2-1);

for i=1:size(tedata,2)  % test data
        tedata(:,i) = tedata(:,i) - MItrain(i);
        if MAtrain(i)~=0
            tedata(:,i) = tedata(:,i) / MAtrain(i);
        else
            tedata(:,i) = 0.5.*ones(size(tedata,1),1);
        end
end
tedata = double( tedata*2-1);