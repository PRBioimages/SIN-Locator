function prelabels = trainSVMmodel(trdata, tedata, target, baMethod)
% use SVM model
% trdata: training data. n*d matrix. Normalized feature matrix. n is the number of samples
% and d is the number of features
% target: 1*n vector. Elements are 1, 2, 3,... (class index)
% tedata: testing data.

if size(target,2)~=1
    target = target';
end

% sampling to solve data imbalance
u = unique(target); % 
[trdata, target] = balanceData(trdata, target, u, baMethod);

% search model parameters
c = 2.^[-5:10];
g = 2.^[5:-1:-14];
acc_cv = zeros(length(c),length(g));
for m=1:length(c)
    for n = 1:length(g)
        options = ['-v 10 -c ' num2str(c(m)) ' -g ' num2str(g(n)) ' -b 0'];
        acc_cv(m,n) = svmtrain( target, trdata, options);
    end
end
% select the best c and g
%     [a, row] = max(acc_cv); 
%     [a, col] = max(a);
%     row = row(col);

[a, row] = max(acc_cv);
bestacc = max(a);
bestcoord = [];
 for m=1:length(c)
    for n = 1:length(g)
        if acc_cv(m,n)>=bestacc
            bestcoord = [bestcoord; [m n]];
        end
    end
 end
row = round(mean(bestcoord(:,1)));
col = round(mean(bestcoord(:,2)));
if acc_cv(row, col)<bestacc
    searchrange = [-1 0; -1 1; 0 1; 1 1; 1 0;  1 -1; 0,-1; -1 -1];  % search for other combinations near [row, col]
    r = 0;  flagr = 0; flagc = 0;
    while acc_cv(row, col)<bestacc
        row_ori = row;
        col_ori = col;
        r = r+1;
        for j = 1:size(searchrange,1)
            if row_ori+r*searchrange(j,1)>=1 && row_ori+r*searchrange(j,1)<=length(c)
                row = row_ori+r*searchrange(j,1);
            else
                flagr = 1;
            end
            if col_ori+r*searchrange(j,2)>=1 && col_ori+r*searchrange(j,2)<=length(g)
                col = col_ori+r*searchrange(j,2);
            else
                flagc = 1;
            end
            if acc_cv(row, col)>=bestacc
                flagr = 1; flagc=1;
                break;
            end
        end
        if flagr == 1 && flagc==1
            break;
        end
    end
end

str_opt= ['-c ' num2str(c(row)) ' -g ' num2str(g(col)) ' -b 1'];
% train model
modelbr = svmtrain(target, trdata, str_opt); 
% test model
[~, ~, dec_values] = svmpredict(ones(size(tedata,1),1), tedata, modelbr, '-b 1');

[~, ind] = sort(modelbr.Label);
prelabels = dec_values(:, ind);