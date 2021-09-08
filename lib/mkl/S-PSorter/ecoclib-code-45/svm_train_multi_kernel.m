function sw = svm_train_multi_kernel(data1,data2,MyClassifparams)
idx = MyClassifparams.idx;
idfeat = MyClassifparams.idfeat;
gamma = MyClassifparams.gamma;
C = MyClassifparams.C;
indcom = MyClassifparams.indcom;
betasknown = MyClassifparams.betasknown;

kfold=5;
traindata=[data1;data2];
gnd=[ones(size(data1,1),1);-ones(size(data2,1),1)];  % 0->1,  1->-1

for i = 1:length(gamma) % for each feature type
    trdata = traindata(:,idx{1,i});
    MK(:,:,i)=calckernel('rbf', gamma(i), trdata);
    sw.traindata{i,1} = trdata;
end

if ~isempty(indcom) %
    MKupdate = zeros(size(MK,1), size(MK,2), length(idfeat));
    for i = 1:size(indcom,1)
        for j = 1:size(indcom,2)
            MKupdate(:,:,i) = MKupdate(:,:,i) + betasknown(i,j)*MK(:,:,indcom(i,j));%betasknown; 
        end
    end
    singleInd = setdiff(1:size(MK,3), reshape(indcom,1,[]));
    for j = 1:length(singleInd)
        MKupdate(:,:,i+j) = MK(:,:,singleInd(j));
    end
else
    MKupdate = MK;
end

sw.result = jb_2gridSearch(MKupdate, gnd, kfold, C);