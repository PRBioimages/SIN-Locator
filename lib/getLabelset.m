function [preTable,cons,sita] = getLabelset(outputs,criway,sitaway,cons,sita, indeTarget)
% get labels from scores
switch criway
    case 0 
        % single-label classification
        preTable = zeros(1, size(outputs,1));
        for i = 1:size(outputs,1)
            output = outputs(i,:);
            [~,index]=max(output);
            preTable(1,i) = index;
        end
    case 1
        % Top criterion
         preTable = -ones(size(outputs));
         for i = 1:size(outputs,2)
            output = outputs(:,i);
            [~,index]=max(output);
             % Top criterion
             if(max(output) <= 0) 
                 preTable(index,i) = 1;
             else
                 preTable(output>0,i) = 1;
             end 
         end
         
    case 2
        % sita criterion
        switch sitaway
            case 1
                % way1: 
                cons = 0.5;
                sita = 0.2;      
            case 2
                % way2: the model.cons model.sita
                cons = sum(cons)/length(cons);
                sita = sum(sita)/length(sita);
            case 3
                % way3: grid search using the outputs
                [cons,sita] = gridSearchSita(outputs,indeTarget);
        end
        preTable = zeros(size(outputs));
         for i = 1:size(outputs,2)
            output = outputs(:,i);
            [maxi,index]=max(output);
             if(max(output) <= 0) 
                 preTable(index,i) = 1;
             else
                 preTable(output>cons,i) = 1;
                 preTable(output>=maxi*sita,i) = 1;
             end 
         end
end

function [cons,sita] = gridSearchSita(outputs,indeTarget)
conss = 0:0.01:1;
sitas =0:0.01:1;
re = zeros(length(conss), length(sitas));
for c = 1:length(conss)
    for s=1:length(sitas)
        % for each combination
        co = conss(c);
        si = sitas(s);
        preTable = zeros(size(outputs));
        pAll = [];
        for i = 1:size(outputs,2) % for each sample
             % get predicted labels
            output = outputs(:,i);
            [maxi,index]=max(output);
             if(max(output) <= 0) 
                 preTable(index,i) = 1;
             else
                 preTable(output>co,i) = 1;
                 preTable(output>=maxi*si,i) = 1;
             end 
             % compare with target
             p = 0;
             for j=1:size(outputs,1)
                 if preTable(j,i)==indeTarget(j,i)
                     p = p+1;
                 end
             end
             p = p/size(outputs,1);
             pAll = [pAll; p];
         end
         %
         re(c,s) = mean(pAll);
    end
end
[a, row] = max(re); 
[a, col] = max(a);
row = row(col);
cons = conss(row);
sita = sitas(col);



    