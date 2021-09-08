function ensembleAccuracy=ensembleStrategy(label,predictLabel)
db1=[];db2=[];db3=[];db4=[];db5=[];db6=[];db7=[];db8=[];db9=[];db10=[];
Label=[];
for j=1:5
Label=[Label;label{j}]; 
db1=[db1;predictLabel{1,j}'];
db2=[db2;predictLabel{2,j}'];
db3=[db3;predictLabel{3,j}'];
db4=[db4;predictLabel{4,j}'];
db5=[db5;predictLabel{5,j}'];
db6=[db6;predictLabel{6,j}'];
db7=[db7;predictLabel{7,j}'];
db8=[db8;predictLabel{8,j}'];
db9=[db9;predictLabel{9,j}'];
db10=[db10;predictLabel{10,j}'];
% label=[label;Kappa_Labels_Forest{j}];
end
db=[db1,db2,db3,db4,db5,db6,db7,db8,db9,db10];
for i=1:10
%    Accuracy(i)=length(find(Mymode(db,i)==Label'))/length(Label);
    Accuracy(i)=length(find(mode(db')==Label'))/length(Label);
end
ensembleAccuracy=max(Accuracy);