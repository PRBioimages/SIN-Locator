function para = getparas(str, repeatNumNode2vec)
% set parameters
switch str
    case 'featTypeIFimg'
        para = {'IF_perPro_aveFeats';... 
            'IF_perPro_aveNanniLF';...
            
            'IF_perPro_AlexnetFeats';...
            'IF_perPro_vgg19netFeats';...
            'IF_perPro_googlenetFeats';...
            'IF_perPro_resnet18Feats';...
            'IF_perPro_resnet50Feats';...
            'IF_perPro_resnet101Feats';...
            'IF_perPro_inceptionv3netFeats';...
            'IF_perPro_inceptionresnetv2netFeats';...
            'IF_perPro_squeezenetFeats';...
            
            'IF_perPro_bestFitting512LastLayerFeats';...
            'IF_perPro_bestFitting512PenultimateLayerFeats';...
            'IF_perPro_bestFitting1024LastLayerFeats';...
            'IF_perPro_bestFitting1024PenultimateLayerFeats'...
            };

    case 'featTypeIHCimg'
         para = {'IHC_perPro_aveFeats_wholeImage';...
            'IHC_perPro_aveNanniLF_wholeImage';...
            'IHC_perPro_AlexnetFeats_wholeImage';...
            'IHC_perPro_vgg19netFeats_wholeImage';...
            'IHC_perPro_googlenetFeats_wholeImage';...
            'IHC_perPro_resnet18Feats_wholeImage';...
            'IHC_perPro_resnet50Feats_wholeImage';...
            'IHC_perPro_resnet101Feats_wholeImage';...
            'IHC_perPro_inceptionv3netFeats_wholeImage';...
            'IHC_perPro_inceptionresnetv2netFeats_wholeImage';...
            'IHC_perPro_squeezenetFeats_wholeImage';...
            
            'IHC_perPro_aveFeats_patchSize227';...
            'IHC_perPro_aveNanniLF_patchSize227';...

            'IHC_perPro_AlexnetFeats_patchSize227';...
            'IHC_perPro_vgg19netFeats_patchSize227';...
            'IHC_perPro_googlenetFeats_patchSize227';...
            'IHC_perPro_resnet18Feats_patchSize227';...
            'IHC_perPro_resnet50Feats_patchSize227';...
            'IHC_perPro_resnet101Feats_patchSize227';...
            'IHC_perPro_inceptionv3netFeats_patchSize227';...
            'IHC_perPro_inceptionresnetv2netFeats_patchSize227';...
            'IHC_perPro_squeezenetFeats_patchSize227'...
            };
    case 'featTypeNode2vec'
        para = {'IF_perPro_node2vec500wFeats';...
            'IHC_perPro_node2vec500wFeats'};  

end