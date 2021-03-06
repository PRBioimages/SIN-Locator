clear all
clc
close all


currentFolder = pwd;
addpath(genpath(currentFolder))
load idx_sda.mat
for i=1:10
      fileName=strcat(num2str(i),'.mat');
      load(fileName);
    for j=1:5
        idx_sda=A{i}{j}
        temp_result_multi_kernel_structure=0;
        temp_result_multi_kernel_forest=0;
        temp_result_multi_kernel=0;
        [trainSet,testSet]=jb_kfold(labels,5,j);
        trainLabels=labels(trainSet);
        testLabels=labels(testSet);
        traindata=data(trainSet,:);
        testdata=data(testSet,:);

        [train_data,test_data] = featnorm(traindata,testdata);
        train_data = double(train_data*2-1);
        test_data = double(test_data*2-1);
        for m=0.9:0.1:2.1
              for n=0.9:0.1:2.1
                gamma=[m,n];
                base_params.idx_sda=idx_sda;
                base_params.gamma=gamma;
               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%S-PSorter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5                

                Parameters.coding='OneVsAll';
                Parameters.base='svm_train_multi_kernel';
                Parameters.base_test='svm_test_multi_kernel';
                Parameters.base_test_params= base_params;
                Parameters.base_params=base_params;
                [Classifiers,Parameters]=ECOCTrain(train_data,trainLabels,Parameters);
                [X,Labels,Values2,confusion]=ECOCTest(test_data,Classifiers,Parameters,testLabels)
                result_multi_kernel(i,j)=sum(diag(confusion))/sum(sum(confusion));
                if  result_multi_kernel(i,j)> temp_result_multi_kernel
                   temp_result_multi_kernel=result_multi_kernel(i,j);
                   multi_kernel_predict_labels{i,j}=Labels;
                   multi_kernel_labels{j}=testLabels;
             else
                   result_multi_kernel(i,j)= temp_result_multi_kernel;
             end 

              end

        end          
      end
end
ensemble_OVAPSorter=ensembleStrategy(multi_kernel_labels,multi_kernel_predict_labels);

