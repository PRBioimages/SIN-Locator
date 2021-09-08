function [model]=svm_train_single_kernel(data1,data2,MyClassifparams)
data=[data1;data2];
labels=[ones(size(data1,1),1);-ones(size(data2,1),1)];
model = svmtrain(labels,data,MyClassifparams);
[pred_label,accuracy,p]=svmpredict(labels,data,model);
accuracy
