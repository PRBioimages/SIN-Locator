function [pred_label,p]=svm_test_single_kernel(data,classifier,MyClassifparams)
[pred_label,accuracy,p]=svmpredict(ones(size(data,1),1),data,classifier);
