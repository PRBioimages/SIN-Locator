clear all
clc
close all


currentFolder = pwd;
addpath(genpath(currentFolder))
load idx_sda.mat
for i=1:10 %db1-db10
      fileName=strcat(num2str(i),'.mat');
      load(fileName);
    for j=1:5 % 5fold cross validation
        idx_sda=A{i}{j}
        temp_result_multi_kernel_structure=0;
        temp_result_multi_kernel_forest=0;
        temp_result_multi_kernel=0;
        [trainSet,testSet]=jb_kfold(labels,5,j);
        trainLabels=labels(trainSet);
        testLabels=labels(testSet);
        traindata=data(trainSet,:);
        testdata=data(testSet,:);

        %[train_data,test_data] = featnorm(traindata,testdata);
        train_data = double(traindata*2-1);
        test_data = double(testdata*2-1);
        for m=0.9:0.1:2.1 % gamma for data1 (haralick features)
              for n=0.9:0.1:2.1 % gamma for data2 (LBP features)
                gamma=[m,n];
                base_params.idx_sda=idx_sda;
                base_params.gamma=gamma;
               
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%S-PSorter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5                
                tic
                Parameters.coding='CUSTOM';
                Parameters.custom_coding='structure';
                Parameters.base='svm_train_multi_kernel';
                Parameters.base_test='svm_test_multi_kernel';
                Parameters.base_test_params= base_params;
                Parameters.base_params=base_params;
                [Classifiers,Parameters]=ECOCTrain(train_data,trainLabels,Parameters);
                [X,Labels,temp_Values,confusion]=ECOCTest(test_data,Classifiers,Parameters,testLabels);
                result_multi_kernel_structure(i,j)=sum(diag(confusion))/sum(sum(confusion));
                if  result_multi_kernel_structure(i,j)> temp_result_multi_kernel_structure
                    temp_result_multi_kernel_structure=result_multi_kernel_structure(i,j);
                    multi_kernel_predict_labels_struture{i,j}=Labels;
                    multi_kernel_labels_structure{j}=testLabels;
                else
                   result_multi_kernel_structure(i,j)= temp_result_multi_kernel_structure;
                end
                toc
              end

        end          
      end
end
ensemble_SPSorter=ensembleStrategy(multi_kernel_labels_structure,multi_kernel_predict_labels_struture);

