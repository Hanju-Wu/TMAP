thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

% Use 10 instances per sparsity level. Each method runs once per instance;
% all requested gaps are extracted from that same trajectory.
numInstances = 10;
rhos = [0.01, 0.05, 0.1];
tolerances = [1e-4, 1e-8, 1e-12]; % relative objective gaps
stopCriterion = 'relativeFunctionGap';

outputFolder = fullfile(thisDir,'output');
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
allSummary = table();
allMetrics = table();

cleanup = onCleanup(@clearGaussianEnvironment);
for rho = rhos
    fprintf('\n===== Gaussian reconstruction: rho=%.2f, final gap=%.0e =====\n', ...
        rho,min(tolerances));
    setenv('LASSO_RHO',sprintf('%.17g',rho));
    setenv('LASSO_TOLS',sprintf('%.17g ',tolerances));
    setenv('LASSO_INSTANCES',sprintf('%d',numInstances));
    setenv('LASSO_STOP_CRITERION',stopCriterion);
    run(fullfile(thisDir,'Example.m'));

    caseSummary = addvars(summaryTable,repmat(rho,height(summaryTable),1), ...
        'Before',1,'NewVariableNames',{'rho'});
    allSummary = [allSummary;caseSummary]; %#ok<AGROW>
    caseMetrics = addvars(metricTable,repmat(rho,height(metricTable),1), ...
        'Before',1,'NewVariableNames',{'rho'});
    allMetrics = [allMetrics;caseMetrics]; %#ok<AGROW>
    writetable(allSummary,fullfile(outputFolder, ...
        'gaussian-n2^14-all-tolerances-summary.csv'));
    writetable(allMetrics,fullfile(outputFolder, ...
        'gaussian-n2^14-all-tolerances-instance-metrics.csv'));
end
clear cleanup;

function clearGaussianEnvironment()
setenv('LASSO_RHO','');
setenv('LASSO_TOL','');
setenv('LASSO_TOLS','');
setenv('LASSO_INSTANCES','');
setenv('LASSO_STOP_CRITERION','');
end
