

% initialize
addpath ./lib
addpath ./lib/libsvm-3.24/matlab
addpath(genpath('./lib/mkl/S-PSorter'))

if ~exist('./data','dir')
    mkdir('./data');
end

if ~exist('./tmp','dir')
    mkdir('./tmp');
end

if ~exist('./data/1_IFimages','dir')
    mkdir('./data/1_IFimages');
end

if ~exist('./data/2_IHCimages','dir')
    mkdir('./data/2_IHCimages');
end

if ~exist('./data/3_sequences','dir')
    mkdir('./data/3_sequences');
end

if ~exist('./data/4_IFfeats/masks','dir')
    mkdir('./data/4_IFfeats/masks');
end

if ~exist('./data/5_IHCfeats','dir')
    mkdir('./data/5_IHCfeats');
end

if ~exist('./data/6_sequenceFeats','dir')
    mkdir('./data/6_sequenceFeats');
end

if ~exist('./data/7_constructedData','dir')
    mkdir('./data/7_constructedData');
end

if ~exist('./data/8_classification/sdalog','dir')
    mkdir('./data/8_classification/sdalog');
end

