function [result]=jb_2gridSearch(allK,gnd,kfold,C, tcv,fcv)
% input
%   gnd: class label.
%   kfold: the number of fold.
%   tcv: training set.  for partition.
%   fcv: test set
% output
%   the weight .
parak='-t 4';
flag=0;
if(nargin<5)
    flag=1;
end
numFeats = size(allK,3);
d = 0.02; %d = 0.02; %%%%%%%%%%%%%%
ds = 0:d:1;  %  0 0.1 0.2 ... 1
betas = generateBetas(ds, numFeats);
if size(betas,1)>2000
    betas = betas(randi(size(betas,1),1,2000),:);
end

maxacc=-1;  tmppos = [];
accSet = [];  aucSet = [];
for i = 1:size(betas,1) % for each beta combination
    i
    icc=0;
    truel=[];
    restmp=[];
    p=[];
    for ttcc=1:kfold
        if(flag==1)
            [tr,te]=jb_kfold(gnd,kfold,ttcc);
        else
            tr=tcv{ttcc}';
            te=fcv{ttcc}'; 
        end
        nte=length(te);
        [a,b,c] = jb_classfyWithGramMatrixLinearMKL(allK, gnd, tr, te, betas(i,:), C);                                                                             
        restmp(ttcc,1)=a;
        prel=b;
        p(icc+1:icc+nte)=c;
        truel(icc+1:icc+nte,1)=gnd(te,:); 
        icc=icc+nte;
    end
    tmp=mean(restmp);                                
    [xc,yc,T,auc]=perfcurve(truel,-p',1); 
    [accSet, aucSet] = geneSet(accSet, aucSet, betas(i,:), tmp, auc);
    if(tmp>maxacc)
        maxacc=tmp;
        tmppos = betas(i,:);
    end
end
    
beta=tmppos;
K=allK;
tK=K(:,:,1)*beta(1);
for ii=2:length(beta)
    tK=tK+K(:,:,ii)*beta(ii);
end
model = svmtrain(gnd, [(1:length(gnd))', tK], parak);
[pred_label,accuracy,p] = svmpredict(gnd,[(1:length(gnd))', tK], model);
result.model=model;
result.beta=beta;
        

function betas = generateBetas(ds, numFeats)
% generate betas
betas = [];
switch numFeats % number of feature types
    case 1
        betas = 1;
    case 2
        for i = 1:length(ds)
            betas(i,1) = ds(1,i);
        end
        betas = [betas 1-sum(betas,2)];
    case 3
        for i = 1:length(ds)
            for j = 1:length(ds)
                com = [ds(1,i) ds(1,j)];
                if sum(com)<=1
                    betas = [betas; com];
                end
            end
        end
        betas = [betas 1-sum(betas,2)];
    case 4
        for i = 1:length(ds)
            for j = 1:length(ds)
                for k = 1:length(ds)
                    com = [ds(1,i) ds(1,j) ds(1,k)];
                    if sum(com)<=1
                        betas = [betas; com];
                    end
                end
            end
        end
        betas = [betas 1-sum(betas,2)];
    case 5
        for i = 1:length(ds)
            for j = 1:length(ds)
                for k = 1:length(ds)
                    for l = 1:length(ds)
                        com = [ds(1,i) ds(1,j) ds(1,k) ds(1,l)];
                        if sum(com)<=1
                            betas = [betas; com];
                        end
                    end
                end
            end
        end
        betas = [betas 1-sum(betas,2)];
    case 6
        for i = 1:length(ds)
            for j = 1:length(ds)
                for k = 1:length(ds)
                    for l = 1:length(ds)
                        for m = 1:length(ds)
                            com = [ds(1,i) ds(1,j) ds(1,k) ds(1,l) ds(1,m)];
                            if sum(com)<=1
                                betas = [betas; com];
                            end
                        end
                    end
                end
            end
        end
        betas = [betas 1-sum(betas,2)];
end
 

function [accSet, aucSet] = geneSet(accSet, aucSet, betas_i, tmp, auc)

betas_i = round(betas_i*10+1);

switch size(betas_i,2)
    case 1
        accSet(betas_i) =  tmp;
        aucSet(betas_i) = auc;
    case 2
        accSet(betas_i(1,1), betas_i(1,2)) = tmp;
        aucSet(betas_i(1,1), betas_i(1,2)) = auc;
    case 3
        accSet(betas_i(1,1), betas_i(1,2), betas_i(1,3)) = tmp;
        aucSet(betas_i(1,1), betas_i(1,2), betas_i(1,3)) = auc;
    case 4
        accSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4)) = tmp;
        aucSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4)) = auc;
    case 5
        accSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4), betas_i(1,5)) = tmp;
        aucSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4), betas_i(1,5)) = auc;
    case 6
        accSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4), betas_i(1,5), betas_i(1,6)) = tmp;
        aucSet(betas_i(1,1), betas_i(1,2), betas_i(1,3), betas_i(1,4), betas_i(1,5), betas_i(1,6)) = auc;
end
        
        
        
        
        
        
        
        
        
