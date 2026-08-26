function prepare_ssnal_datasets
%PREPARE_SSNAL_DATASETS Reproduce polynomial Lasso data sets from SSNAL.

thisDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(thisDir,'dataset');
compiledDir = fullfile(thisDir,'L1GeneralExamples','minFunc_2012','compiled');
addpath(compiledDir);

problems = struct( ...
    'source', {'abalone_scale','housing_scale','mpg_scale','space_ga_scale', ...
        'bodyfat_scale'}, ...
    'output', {'abalone7.mat','housing7.mat','mpg7.mat','space_ga9.mat', ...
        'bodyfat7.mat'}, ...
    'degree', {7,7,7,9,7}, ...
    'expectedRows', {4177,506,392,3107,252}, ...
    'expectedCols', {6435,77520,3432,5005,116280}, ...
    'paperLambdaMax', {5.21e5,3.28e5,1.28e4,4.01e3,5.29e4});

for problemIndex = 1:numel(problems)
    problem = problems(problemIndex);
    sourceFile = fullfile(dataDir,problem.source);
    outputFile = fullfile(dataDir,problem.output);
    fprintf('\nGenerating %s from %s (degree %d)...\n', ...
        problem.output,problem.source,problem.degree);

    [b,X] = libsvmread(sourceFile);
    X = full(X);
    nonzeroColumns = any(X ~= 0,1);
    X = X(:,nonzeroColumns);
    [sampleCount,featureCount] = size(X);
    exponents = polynomialExponents(featureCount,problem.degree);
    expandedCount = size(exponents,1);

    assert(sampleCount == problem.expectedRows, ...
        'Unexpected row count for %s.',problem.source);
    assert(expandedCount == problem.expectedCols, ...
        'Unexpected expanded column count for %s.',problem.source);

    A = zeros(sampleCount,expandedCount);
    powers = cell(featureCount,1);
    for featureIndex = 1:featureCount
        powers{featureIndex} = X(:,featureIndex).^(0:problem.degree);
    end

    blockSize = 512;
    for blockStart = 1:blockSize:expandedCount
        blockEnd = min(blockStart + blockSize - 1,expandedCount);
        blockExponents = exponents(blockStart:blockEnd,:);
        block = ones(sampleCount,blockEnd - blockStart + 1);
        for featureIndex = 1:featureCount
            featureExponents = double(blockExponents(:,featureIndex));
            for power = 1:problem.degree
                selected = featureExponents == power;
                if any(selected)
                    block(:,selected) = block(:,selected) .* ...
                        powers{featureIndex}(:,power + 1);
                end
            end
        end
        A(:,blockStart:blockEnd) = block;
        fprintf('  columns %d-%d of %d\r',blockStart,blockEnd,expandedCount);
    end
    fprintf('\n');

    lambdaMaxEstimate = normest(A)^2;
    relativeSpectralDifference = abs(lambdaMaxEstimate - ...
        problem.paperLambdaMax)/problem.paperLambdaMax;
    assert(relativeSpectralDifference < 0.05, ...
        ['Spectral check failed for %s: estimate %.4e, paper %.4e. ' ...
         'The source or polynomial expansion is inconsistent.'], ...
        problem.source,lambdaMaxEstimate,problem.paperLambdaMax);
    metadata = struct( ...
        'source',problem.source, ...
        'degree',problem.degree, ...
        'originalFeatureCount',featureCount, ...
        'expandedFeatureCount',expandedCount, ...
        'includesConstantFeature',true, ...
        'removedAllZeroFeatures',find(~nonzeroColumns), ...
        'lambdaMaxEstimate',lambdaMaxEstimate, ...
        'paperLambdaMax',problem.paperLambdaMax, ...
        'relativeSpectralDifference',relativeSpectralDifference, ...
        'sourceUrl',['https://www.csie.ntu.edu.tw/~cjlin/' ...
            'libsvmtools/datasets/regression/' problem.source]);

    save(outputFile,'A','b','metadata','-v7.3');
    fileInfo = dir(outputFile);
    assert(fileInfo.bytes < 1024^3, ...
        'Generated file %s exceeds the 1 GB limit.',problem.output);
    fprintf('Saved %s: %d-by-%d, %.1f MB, lambda_max %.4e ', ...
        problem.output,sampleCount,expandedCount, ...
        fileInfo.bytes/1024^2,lambdaMaxEstimate);
    fprintf('(paper %.4e, relative difference %.2e).\n', ...
        problem.paperLambdaMax,relativeSpectralDifference);
end
end

function exponents = polynomialExponents(featureCount,maxDegree)
% Rows enumerate all monomials of total degree at most maxDegree.
expandedCount = nchoosek(featureCount + maxDegree,maxDegree);
exponents = zeros(expandedCount,featureCount,'uint8');
rowStart = 1;
for degree = 0:maxDegree
    degreeExponents = weakCompositions(degree,featureCount);
    rowEnd = rowStart + size(degreeExponents,1) - 1;
    exponents(rowStart:rowEnd,:) = degreeExponents;
    rowStart = rowEnd + 1;
end
end

function compositions = weakCompositions(total,partCount)
compositionCount = nchoosek(total + partCount - 1,partCount - 1);
compositions = zeros(compositionCount,partCount,'uint8');
current = zeros(1,partCount,'uint8');
nextRow = 1;
fillPart(1,total);

    function fillPart(partIndex,remaining)
        if partIndex == partCount
            current(partIndex) = remaining;
            compositions(nextRow,:) = current;
            nextRow = nextRow + 1;
            return;
        end
        for value = 0:remaining
            current(partIndex) = value;
            fillPart(partIndex + 1,remaining - value);
        end
    end
end
