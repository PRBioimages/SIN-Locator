function evalCriteria = evalModel(outputs, preTable, testTarget, multiFlag)
% evaluate the results
% inputs: outputs are predicted scores. It is a L*n matirx, where L is the number of classes, and
%            n is the number of samples.
%            preTable represent predicted labels. It a L*n matirx. Elements in the matrix are 0 or 1. 
%            testTarget is ground truth. It is also a L*n matrix.  Elements in the matrix are 0 or 1. 
%            multiFlag represents whether the task is a multi-label classification.

if multiFlag ==0
    % single-label classification evaluation
    evalCriteria.subset_accuracy = S_accuracy(preTable,testTarget);
    [evalCriteria.confusionMat,  evalCriteria.accuracy_s, evalCriteria.sensitivity,...
        evalCriteria.specificity, evalCriteria.precision, evalCriteria.recall,...
        evalCriteria.Fscore] = confusionmatStats(testTarget, preTable);
else
    % multi-label classification evaluation
    evalCriteria.subset_accuracy = S_accuracy(preTable,testTarget);
    [evalCriteria.accuracy,evalCriteria.recall,evalCriteria.precision,evalCriteria.F1_score] = Accuracy(preTable,testTarget);
    [evalCriteria.label_accuracy, evalCriteria.average_label_accuracy] = L_accuracy(preTable,testTarget);
    evalCriteria.hloss = Hamming_loss(preTable,testTarget);
    evalCriteria.one_error = One_error(outputs,preTable,testTarget);
    evalCriteria.coverage = Coverage(outputs,testTarget);
    evalCriteria.rloss = Ranking_loss(outputs,preTable,testTarget);
    evalCriteria.avgprec = Average_precision(outputs,preTable,testTarget);
    evalCriteria.confusionMatrix = ConfusionMatrix(preTable,testTarget);
end

function subset_accuracy = S_accuracy(prediction,testtarget)
% Caculate the Subset accuracy of testing.
if size(prediction)~=size(testtarget)
    error( 'The sizes of prediction and turth are not match');
else
    num_test= size(testtarget,2);
    num=0;
    for i=1:num_test
       if sum(prediction(:,i)~=testtarget(:,i))==0
          num=num+1;
       end  
    end
    subset_accuracy = num/num_test;
end

function [confusionMat,  accuracy_s, sensitivity, specificity, precision, recall,...
    Fscore] = confusionmatStats(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 

if nargin < 2
    confusionMat = group;
else
    confusionMat = confusionmat(group,grouphat);
end
value1 = confusionMat;

numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));
    
accuracy_s = (2*trace(value1)+sum(sum(2*value1)))/(numOfClasses*totalSamples);

[TP,TN,FP,FN,sensitivity,specificity,precision,Fscore] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   FN(class) = sum(value1(class,:))-TP(class);
end

for class = 1:numOfClasses
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    Fscore(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end
recall = sensitivity;


function [accuracy,recall,precision,F1_score]=Accuracy(prediction,testtarget)
% Caculate the Accuracy of testing. 
if size(prediction)~=size(testtarget)
     error( 'The sizes of prediction and turth are not match');
else
     [num_class,num_test]= size(testtarget);
     scop=zeros(1,num_test);
     recall_new=zeros(1,num_class);
     precision_new=zeros(1,num_class);
     y_recall=zeros(1,num_class);
     y_precision=zeros(1,num_class);
     for i=1:num_test % for each sample
         tmp1=testtarget(:,i)';
         tmp2=prediction(:,i)';
         mix=0;
         union=0;
         for j=1:length(tmp1) % for each label
             if tmp1(j)==tmp2(j)&&tmp1(j)==1
                mix=mix+1;
                union=union+1;
             end
             if tmp1(j)~=tmp2(j)
                 union=union+1;
             end
         end
         scop(i)=(mix/union); % point

         for j=1:num_class % for each label
             if tmp1(j)==1
                 y_recall(j)=y_recall(j)+1;
                 recall_new(j)=recall_new(j)+scop(i);
             end
             if tmp2(j)==1
                 y_precision(j)=y_precision(j)+1;
                 precision_new(j)=precision_new(j)+scop(i);
             end
         end      
     end
     
     accuracy=sum(scop)/num_test;
     for i=1:num_class
         recall_new(i)=recall_new(i)/y_recall(i);
         precision_new(i)=precision_new(i)/y_precision(i);
     end
     %calculate F1 score
     for i=1:num_class
         if ~isnan(precision_new(i))
             f1_score(i)=2*recall_new(i)*precision_new(i)/(recall_new(i)+precision_new(i));
         end
     end
     F1_score(1,1) = mean(f1_score(:, ~isnan(f1_score)));
     recall = mean(recall_new(:, ~isnan(recall_new)));
     precision = mean(precision_new(:, ~isnan(precision_new)));
     F1_score(1,2) = 2*recall*precision/(recall+precision);

end

function [label_accuracy, average_label_accuracy] = L_accuracy(prediction,testtarget)
% Caculate the Label accuracy and average label accuracy of testing
if size(prediction)~=size(testtarget)
      error( 'The sizes of prediction and turth are not match');
else
      [num_class,num_test]= size(testtarget);
       label_accuracy=zeros(1,num_class);
       for m=1:num_class
           temp1=prediction(m,:);
           temp2=testtarget(m,:);
           temp=temp1-temp2;
           correct=length(find(temp==0));
           accuracy=correct/num_test;
           label_accuracy(m)=accuracy;
       end
       average_label_accuracy = sum(label_accuracy)/num_class;
end

function hloss = Hamming_loss(prediction,testtarget)

if size(prediction)~=size(testtarget)
    error( 'The sizes of prediction and turth are not match');
else
    [num_class,num_test] = size(testtarget);
    loss = 0;
    for i=1:num_test
        a = find(prediction(:,i)==1);
        b = find(testtarget(:,i)==1);
        c = length(a)+length(b)-2*length(intersect(a,b));
        loss = loss + c/num_class;  
    end
    hloss = loss/num_test;
end

function one_error = One_error(outputs,prediction,testtarget)

if size(prediction)~=size(testtarget)
    error( 'The sizes of prediction and turth are not match');
else
    [~,num_test] = size(testtarget);
    err = 0;
    for i=1:num_test
        [~,a] = max(outputs(:,i));
        if testtarget(a,i) == 0
            err = err + 1;  
        end
    end
    one_error = err/num_test;
end

function coverage = Coverage(outputs,testtarget)
%Computing the coverage
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

[num_class,num_instance]=size(outputs);

Label=cell(num_instance,1);
not_Label=cell(num_instance,1);
Label_size=zeros(1,num_instance);
for i=1:num_instance
   temp=testtarget(:,i);
   Label_size(1,i)=sum(temp==ones(num_class,1));
   for j=1:num_class
       if(temp(j)==1)
           Label{i,1}=[Label{i,1},j];
       else
           not_Label{i,1}=[not_Label{i,1},j];
       end
   end
end

cover=0;
for i=1:num_instance
   temp=outputs(:,i);
   [tempvalue,index]=sort(temp);
   temp_min=num_class+1;
   for m=1:Label_size(i)
       [tempvalue,loc]=ismember(Label{i,1}(m),index);
       if(loc<temp_min)
           temp_min=loc;
       end
   end
   cover=cover+(num_class-temp_min+1);
end
coverage=(cover/num_instance)-1;

function rloss = Ranking_loss(outputs,prediction,testtarget)

if size(prediction)~=size(testtarget)
    error( 'The sizes of prediction and turth are not match');
else
    [num_class,num_test] = size(testtarget);
    loss = 0;
    for i=1:num_test
        err = 0;
        a = find(testtarget(:,i)==1);
        b = setdiff(1:num_class,a);
        for j = 1:length(a)
            for k = 1:length(b)
                if outputs(a(j),i)<outputs(b(k),i)
                    err = err + 1;
                end
            end
        end
        
        loss = loss + err/(length(a)*length(b));
    end
    rloss = loss/num_test;
end

function avgprec = Average_precision(outputs,prediction,testtarget)

if size(prediction)~=size(testtarget)
    error( 'The sizes of prediction and turth are not match');
else
    [~,num_test] = size(testtarget);
    err = 0;
    for i=1:num_test
        [~,a] = sort(outputs(:,i),'descend');
        b = find(testtarget(:,i)==1);
        num1 = 0;
        for j = 1:length(b)
            num2 = 0;
            for k = 1:length(b)
                if outputs(b(k),i)>=outputs(b(j),i)
                    num2 = num2 + 1;
                end
            end
            num1 = num1 + num2/find(a==b(j));
        end
        err = err + num1/length(b);
    end
    avgprec = err/num_test;
end

function confusionMatrix = ConfusionMatrix(preTable,testTarget)
% code from paper:
% KRSTINI? D, BRAOVI? M, ?ERI? L, et al. Multi-label classifier performance evaluation with confusion matrix [J]. Computer Science & Information Technology, 2020, 10(8): 1-14. 

if size(preTable)~=size(testTarget)
    error( 'The sizes of prediction and turth are not match');
else
    [num_class,num_test] = size(testTarget);
    confusionMatrix = zeros(num_class,num_class);
    for i = 1:num_test
        C_i = zeros(num_class,num_class);
        t1 = testTarget(:,i); % groundtruth Y
        t2 = preTable(:,i); % prediction Z
        tmp1_2 = t1-t2;   tmp1_2(tmp1_2==-1)=0;  % Y\Z
        tmp2_1 = t2-t1;   tmp2_1(tmp2_1==-1)=0; % Z\Y
        if sum(t1~=t2)==0  % fully correct
            C_i = diag(t1);
        else
            if sum(tmp1_2)==0
                tmp = t1.*t2;
                C_i = (tmp*tmp2_1' + sum(t1).*diag(t1))./sum(t2);
            elseif sum(tmp2_1)==0
                C_i = (tmp1_2*t2')./sum(t2) +diag(t2);
            else
                tmp = t1.*t2;
                C_i =  (tmp1_2*tmp2_1')/sum(tmp2_1) +diag(tmp);
            end  
        end
        confusionMatrix = confusionMatrix +C_i;
    end
end


