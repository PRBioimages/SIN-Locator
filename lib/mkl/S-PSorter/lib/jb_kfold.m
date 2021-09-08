function [trainSet,testSet]=jb_kfold(target,kfold,ki)
% K-fold select the training and testing set from set of sample

% para:
%    target: the label of dataset, column vector.
%    kflod:  the number of fold.
%    ki:     the index of current fold.
trainSet=[];
testSet=[];

labelSet=unique(target);
num_class=length(labelSet);
train_lab=[];
test_lab=[];

flag=0;  % 1: leave one out.
for i=1:num_class
    num_ci=length(find(target==labelSet(i)));
    if (num_ci<kfold)
        flag=1;
        break;
    end
end

if(flag==1)
    trainSet=[1:ki-1 ki+1:length(target)]';%tcv{cc}';
    testSet=ki;%fcv{cc}';  
    
else
    for i=1:num_class
        ci=labelSet(i);
        ind=find(target==ci);
        num_ci=length(ind);
        %if (num_ci<kfold)
        %    fprintf(strcat('error! the number of class ',num2str(ci),' is :',num2str(num_ci),',it must be more than kfold:',num2str(kfold),'\n'));
        %    return ;
        %end 
        kk=floor(num_ci/kfold); % the number in each kfold.      
        test_lab=[test_lab ;ind([((ki-1)*kk+1):ki*kk])];
        train_lab=[train_lab; ind([1:(ki-1)*kk (ki*kk+1):end])];
    end

    trainSet=sort(train_lab);
    testSet=sort(test_lab);
end
end