function [res,pred_label,p]=jb_classfyWithGramMatrixLinearMKL(allK,rgnd,trLab,teLab,beta,C) 
nk=size(allK,3);
if(nargin<5)  
   beta=ones(nk)./nk;
end

if (nargin<6)
    parak='-t 4';
else
    parak=['-t 4 -c ', num2str(C)];
end

Y=rgnd(trLab,:);%Y(Y==2)=-1;
Yt=rgnd(teLab,:);%Yt(Yt==2)=-1;
K=allK(trLab,trLab,:);
KT=allK(teLab,trLab,:);
tK=K(:,:,1)*beta(1);
tKT=KT(:,:,1)*beta(1);
for ii=2:length(beta)
     tK=tK+K(:,:,ii)*beta(ii);
     tKT=tKT+KT(:,:,ii)*beta(ii);
end
model = svmtrain(Y, [(1:length(Y))', tK], parak);
[pred_label, accuracy,p] = svmpredict(Yt, [(1:length(Yt))', tKT], model);    
% [pred_label,accuracy,p] = svmpredict(Y, [(1:length(Y))', tK], model);
res=accuracy(1);
end