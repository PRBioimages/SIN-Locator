clear all
clc
close all


load heart_scale.mat
model = svmtrain(heart_scale_label,heart_scale_inst,'-t 0');
svmpredict(heart_scale_label,heart_scale_inst,model);
