function [pred_label,p]=svm_test_multi_kernel(data,classifier,MyClassifparams)
parak='-t 4';
idx=MyClassifparams.idx;
idfeat = MyClassifparams.idfeat;
gamma=MyClassifparams.gamma;
C=MyClassifparams.C;
indcom = MyClassifparams.indcom;
betasknown = MyClassifparams.betasknown;

Yt=ones(size(data,1),1);
beta=classifier.result.beta;
model=classifier.result.model;

for i = 1:length(idx)
    trdata=classifier.traindata{i};
    tedata = data(:, idx{1,i});
    featdata = [trdata; tedata];
    allK(:,:,i)=calckernel('rbf',gamma(i),featdata); 
end
trLab=[1:size(trdata,1)]';
teLab=[size(trdata,1)+1:size(trdata,1)+size(tedata,1)]';

if ~isempty(indcom) %
    allKupdate = zeros(size(allK,1), size(allK,2), length(idfeat));
    for i = 1:size(indcom,1)
        for j = 1:size(indcom,2)
            allKupdate(:,:,i) = allKupdate(:,:,i) + betasknown(i,j)*allK(:,:,indcom(i,j));
        end
    end
    singleInd = setdiff(1:size(allK,3), reshape(indcom,1,[]));
    for j = 1:length(singleInd)
        allKupdate(:,:,i+j) = allK(:,:,singleInd(j));
    end
else
    allKupdate = allK;
end
        
K=allKupdate(trLab,trLab,:);
KT=allKupdate(teLab,trLab,:);
tK=K(:,:,1)*beta(1);
tKT=KT(:,:,1)*beta(1);
for ii=2:length(beta)
     tK=tK+K(:,:,ii)*beta(ii);
     tKT=tKT+KT(:,:,ii)*beta(ii);
end
Yt=ones(size(tedata,1),1);
[pred_label,accuracy,p] = svmpredict(double(Yt), double([(1:length(Yt))', tKT]), model);
