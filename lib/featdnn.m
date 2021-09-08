function outputs = featdnn(trdata, tedata, trlabel, idx, cType)
% use deep neural networks to run classification (MATLAB R2019a)

% idx
idx_use = [];
for i = 1:size(idx,2)
    for j = 1:size(idx,1)
        idx_use = union(idx_use, idx{j,i});
    end
end
intr = trdata(:, idx_use)';  % numFeat*numSample
outte = tedata(:, idx_use)'; 


% run dnn
numFeat = size(intr,1);
numClass = size(trlabel,1);

switch cType  % 'dnn-3'
    case 'dnn-1'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')
            dropoutLayer(0.2)
            fullyConnectedLayer(200)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(200)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn-2'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')
            dropoutLayer(0.2)
            fullyConnectedLayer(400)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(400)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn' % best
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')   % MATLAB R2021a
            dropoutLayer(0.2)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn-4'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')   % MATLAB R2021a
            dropoutLayer(0.2)
            fullyConnectedLayer(800)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(800)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn-5'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')   % MATLAB R2021a
            dropoutLayer(0.2)
            fullyConnectedLayer(1000)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(1000)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];  
    case 'dnn-6'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')
            dropoutLayer(0.2)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn-7'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')
            dropoutLayer(0.2)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
    case 'dnn-8'
        layers = [
            featureInputLayer(numFeat,'Normalization', 'zscore')
            dropoutLayer(0.2)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(600)
            batchNormalizationLayer
            reluLayer
            dropoutLayer(0.4)
            fullyConnectedLayer(numClass)  % last layer
            regressionLayer];
end

miniBatchSize = 128;
options = trainingOptions('adam', ...
    'MaxEpochs', 100,...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch');%,...
    %'Plots','training-progress', ...
    %'Verbose',false );

net = trainNetwork(intr', trlabel', layers, options);
outputs = predict(net, outte');
outputs = outputs';

