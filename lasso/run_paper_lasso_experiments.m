thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));
resultFolder = fullfile(thisDir,'output','paper-lasso-inner-solvers');
if ~exist(resultFolder,'dir'); mkdir(resultFolder); end

datasets = { ...
    'dataset/E2006_test', 'E2006-test'; ...
    'dataset/E2006_train', 'E2006-train'; ...
    'dataset/space_ga9.mat', 'space-ga9'; ...
    'dataset/mpg7.mat', 'mpg7'};
regularizationFactors = [1e-3,1e-4];
summaryTable = table();

logFile = fullfile(resultFolder,'paper-lasso-experiments.log');
if exist(logFile,'file'); delete(logFile); end
diary(logFile);
cleanup = onCleanup(@() cleanupBatchEnvironment());

for datasetIndex = 1:size(datasets,1)
    for regIndex = 1:numel(regularizationFactors)
        datasetPath = datasets{datasetIndex,1};
        datasetLabel = datasets{datasetIndex,2};
        regFactor = regularizationFactors(regIndex);
        regLabel = sprintf('%.0e',regFactor);
        caseTag = sprintf('%s-reg-%s',datasetLabel,regLabel);

        fprintf('\n===== Starting %s =====\n',caseTag);
        setenv('LASSO_DATASET',datasetPath);
        setenv('LASSO_REG',sprintf('%.17g',regFactor));
        run(fullfile(thisDir,'test.m'));

        caseTable = addvars(comparisonTable, ...
            repmat({datasetLabel},height(comparisonTable),1), ...
            repmat(regFactor,height(comparisonTable),1), ...
            repmat(tau,height(comparisonTable),1), ...
            'Before',1, ...
            'NewVariableNames',{'dataset','regFactor','tau'});
        writetable(caseTable,fullfile(resultFolder,[caseTag '-results.csv']));
        summaryTable = [summaryTable;caseTable]; %#ok<AGROW>
        writetable(summaryTable,fullfile(resultFolder,'summary.csv'));

        generatedFigure = fullfile(outputFolder, ...
            [datasetName '-' regularizationTag '-relative-error.png']);
        copyfile(generatedFigure,fullfile(resultFolder,[caseTag '-relative-error.png']));
        fprintf('===== Finished %s =====\n',caseTag);
        close all;
    end
end

diary off;

function cleanupBatchEnvironment()
setenv('LASSO_DATASET','');
setenv('LASSO_REG','');
diary off;
end
