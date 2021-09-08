# SIN-Locator
code for "Ge-Wang, Min-Qi Xue, Hong-Bin Shen, and Ying-Ying Xu, Learning multi-view patterns of subcellular localization from protein imaging, sequence and network data"   
Contact: Ying-Ying Xu, yyxu@smu.edu.cn
2021/06/08

## 1.datasets
The IHC and IF datasets are saved in ./data/0_datasetFiles/datasets, and the updated data can be found at ./data/0_datasetFiles/applicationDataset

## 2.features
All the features used in this study have been extracted and saved in ./data/7_constructedData folder. 
The code of feature extraction can be found at:
#### (1) SLF features 
       http://murphylab.web.cmu.edu/software/2012_PLoS_ONE_Reannotation/
       http://murphylab.web.cmu.edu/software/2008_JProteomeResearch_HPA/
#### (2) GenP 
       https://github.com/LorisNanni
#### (3) CNNs in Matlab
#### (4) Hum-mPLoc 3.0 
       http://www.csbio.sjtu.edu.cn/bioinf/Hum-mPLoc3/
#### (5) DeepLoc
       https://github.com/JJAlmagro/subcellular_localization
#### (6) node2vec
       http://snap.stanford.edu/node2vec/
       
## 3.classification
Run mainProcess.m to recreate the results in the article. The code package has been tested using Matlab R2021a under Windows 7 in a 64bit architecture.
