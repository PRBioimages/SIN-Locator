% code for "Ge-Wang, Min-Qi Xue, Hong-Bin Shen, and Ying-Ying Xu,...
% Learning multi-view patterns of subcellular localization from protein...
% imaging, sequence and network data"   
% Contact: Ying-Ying Xu, yyxu@smu.edu.cn

% initialize
clear
initialize
cType = 'brsvm'; % 'brsvm', 'vote', 'ECOCmkl', 'dnn'

%% datasets and features

% 1. proteins in IF and IHC datasets
load('./data/0_datasetFiles/datasets.mat');

% 2. feature extraction. Features of our dataset have been saved in ./data/7_constructedData
% code can be found at:
% (1) SLF features 
%       http://murphylab.web.cmu.edu/software/2012_PLoS_ONE_Reannotation/
%       http://murphylab.web.cmu.edu/software/2008_JProteomeResearch_HPA/
% (2) GenP 
%       https://github.com/LorisNanni
% (3) CNNs in Matlab
% (4) Hum-mPLoc 3.0 
%       http://www.csbio.sjtu.edu.cn/bioinf/Hum-mPLoc3/
% (5) DeepLoc
%       https://github.com/JJAlmagro/subcellular_localization
% (6) node2vec
%       http://snap.stanford.edu/node2vec/
          

%% 3. classification
baMethod = 'noba';  % 'noba', 'smote', 'down'
% 3.1 only sequence classification
featTyepSeq = {'IF_seq', 'IHC_seq', 'IF_sequence_deeplocFeats','IHC_sequence_deeplocFeats'};
for i = 1:length(featTyepSeq)
    fType = featTyepSeq{i};  % IF sequence features
    classifyPattern(fType, cType, baMethod, './data/7_constructedData', ...
        './data/8_classification');
end

featTyepSeqCom = {'IF_seq+IF_sequence_deeplocFeats', 'IHC_seq+IHC_sequence_deeplocFeats'};
for i = 1:length(featTyepSeqCom)
    fType = featTyepSeqCom{i};  % IF sequence features
    classifyPattern(fType, cType, 'noba', './data/7_constructedData', ...
        './data/8_classification');
end

%% 3.2 only PPI network classification
% use node2vec features
featTypeNode2vec = getparas('featTypeNode2vec'); % all feat strs
for i = 1:length(featTypeNode2vec)  
    fType = featTypeNode2vec{i};
    classifyPattern(fType, cType, 'noba', './data/7_constructedData', ...
        './data/8_classification');
end


%% 3.3 only IF images classification
featTypeIFimg = getparas('featTypeIFimg'); % all feat strs
for i = 1:length(featTypeIFimg)
    classifyPattern(featTypeIFimg{i}, cType, 'noba', './data/7_constructedData',...
        './data/8_classification');
end
% combination of IF features
% The best combination is 'IF_perPro_aveFeats+IF_perPro_bestFitting512PenultimateLayerFeats'
fTypeIFsup = {'IF_perPro_aveFeats+IF_perPro_bestFitting512PenultimateLayerFeats';...
    
% fTypeIFsup = {'IF_perPro_aveFeats+IF_perPro_aveNanniLF'; ...  
%     'IF_perPro_aveFeats+IF_perPro_resnet50Feats';...  
%     'IF_perPro_aveFeats+IF_perPro_bestFitting512LastLayerFeats';...
%     'IF_perPro_aveFeats+IF_perPro_bestFitting512PenultimateLayerFeats'; 
%     'IF_perPro_aveFeats+IF_perPro_bestFitting1024LastLayerFeats';...
%     'IF_perPro_aveFeats+IF_perPro_bestFitting1024PenultimateLayerFeats';...
%     'IF_perPro_aveNanniLF+IF_perPro_bestFitting512LastLayerFeats';...
%     'IF_perPro_aveNanniLF+IF_perPro_bestFitting512PenultimateLayerFeats';...
%     'IF_perPro_aveNanniLF+IF_perPro_bestFitting1024LastLayerFeats';...
%     'IF_perPro_aveNanniLF+IF_perPro_bestFitting1024PenultimateLayerFeats';...
%     'IF_perPro_aveFeats+IF_perPro_aveNanniLF+IF_perPro_bestFitting512LastLayerFeats';...
%     'IF_perPro_aveFeats+IF_perPro_aveNanniLF+IF_perPro_bestFitting512PenultimateLayerFeats';...
%     'IF_perPro_aveFeats+IF_perPro_aveNanniLF+IF_perPro_bestFitting1024LastLayerFeats';...
 %    'IF_perPro_aveFeats+IF_perPro_aveNanniLF+IF_perPro_bestFitting1024PenultimateLayerFeats'...
};
for i = 1:length(fTypeIFsup)
    classifyPattern(fTypeIFsup{i,1}, cType, 'noba', './data/7_constructedData',...
        './data/8_classification');
end

%% 3.4 only IHC images classification
featTypeIHCimg = getparas('featTypeIHCimg'); % all feat strs
% 3.4.1 use whole images    
fTypesIHC_wholeImage = featTypeIHCimg(1:11,:);
for i = 1:length(fTypesIHC_wholeImage)
    classifyPattern(fTypesIHC_wholeImage{i}, cType, 'noba', './data/7_construct.edData', ...
        './data/8_classification');
end
% 3.4.2 use patches 
% 3.4.2.1 use defined size
sizes = 227;
for i = 1:length(sizes) % for each patch size
    fType = ['IHC_perPro_aveFeats_patchSize' num2str(sizes(i))]; % IHC image aveFeats
    classifyPattern(fType, cType, 'noba', './data/7_constructedData', ...
        './data/8_classification');
    fType = ['IHC_perPro_aveNanniLF_patchSize' num2str(sizes(i))]; % IHC image aveNanniLF
    classifyPattern(fType, cType,  'noba', './data/7_constructedData', ...
        './data/8_classification');
end
% 3.4.2.2 use patch size 227 (CNN)
fTypesIHC_patch = featTypeIHCimg(14:end,:); 
for i = 1:length(fTypesIHC_patch)
    classifyPattern(fTypesIHC_patch{i}, cType, 'noba', './data/7_constructedData', ...
        './data/8_classification');
end

% 3.4.3 combination of IF features
% The best combination is 'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet50Feats_patchSize227'
fTypeIHCsup =  {'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet50Feats_patchSize227'}; 
% fTypeIHCsup =
% {'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage';  %run
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_resnet50Feats_patchSize227';...  %run
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet50Feats_patchSize227';...  %run
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_resnet50Feats_wholeImage';...  %run
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet50Feats_wholeImage';...  %run
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_AlexnetFeats_patchSize227';...
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_AlexnetFeats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_resnet18Feats_patchSize227';...
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet18Feats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_inceptionv3netFeats_patchSize227';...
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_inceptionv3netFeats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_inceptionresnetv2netFeats_patchSize227';...
%     'IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_inceptionresnetv2netFeats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet50Feats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_AlexnetFeats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_resnet18Feats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_inceptionv3netFeats_patchSize227';...
%     'IHC_perPro_aveFeats_wholeImage+IHC_perPro_aveNanniLF_wholeImage+IHC_perPro_inceptionresnetv2netFeats_patchSize227'}; 
for i = 1:length(fTypeIHCsup)
    classifyPattern(fTypeIHCsup{i,1}, cType, 'noba', './data/7_constructedData',...
        './data/8_classification');
end

%% 3.5 use three data types
fTypeFinal = {'IF_seqIF_sequence_deeplocFeats+IF_perPro_aveFeatsIF_perPro_bestFitting512PenultimateLayerFeats';...  % seq+img
     'IF_seqIF_sequence_deeplocFeats+IF_perPro_node2vec500wFeats';... % seq+PPI
     'IF_perPro_aveFeatsIF_perPro_bestFitting512PenultimateLayerFeats+IF_perPro_node2vec500wFeats';... % img+PPI
    'IF_seqIF_sequence_deeplocFeats+IF_perPro_aveFeatsIF_perPro_bestFitting512PenultimateLayerFeats+IF_perPro_node2vec500wFeats';... % seq+img+PPI
    'IHC_seqIHC_sequence_deeplocFeats+IHC_perPro_aveNanniLF_wholeImageIHC_perPro_resnet50Feats_patchSize227';... % seq+img
    'IHC_seqIHC_sequence_deeplocFeats+IHC_perPro_node2vec500wFeats';... % seq+PPI
    'IHC_perPro_aveNanniLF_wholeImageIHC_perPro_resnet50Feats_patchSize227+IHC_perPro_node2vec500wFeats';... % img+PPI
    'IHC_seqIHC_sequence_deeplocFeats+IHC_perPro_aveNanniLF_wholeImageIHC_perPro_resnet50Feats_patchSize227+IHC_perPro_node2vec500wFeats'};  % seq+img+PPI
for i = 1:length(fTypeFinal)
    classifyPattern(fTypeFinal{i,1}, cType, 'noba', './data/7_constructedData',...
            './data/8_classification');
end
