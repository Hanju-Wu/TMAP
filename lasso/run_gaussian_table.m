thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

% Set to 100 for the final paper run.  Ten instances are convenient for
% checking the full 3-by-3 parameter grid before launching the long run.
numInstances = 10;
rhos = [0.01, 0.05, 0.1];
tolerances = [1e-2 1e-4, 1e-6];
stopCriterion = getenv('LASSO_STOP_CRITERION');
if isempty(stopCriterion); stopCriterion = 'proxResidual'; end

outputFolder = fullfile(thisDir,'output');
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
allSummary = table();
allMetrics = table();

cleanup = onCleanup(@clearGaussianEnvironment);
for rho = rhos
    for tol = tolerances
        fprintf('\n===== Gaussian reconstruction: rho=%.2f, tol=%.0e =====\n',rho,tol);
        setenv('LASSO_RHO',sprintf('%.17g',rho));
        setenv('LASSO_TOL',sprintf('%.17g',tol));
        setenv('LASSO_INSTANCES',sprintf('%d',numInstances));
        setenv('LASSO_STOP_CRITERION',stopCriterion);
        run(fullfile(thisDir,'Example.m'));

        caseSummary = summaryTable;
        caseSummary = addvars(caseSummary, ...
            repmat(rho,height(caseSummary),1), ...
            repmat(tol,height(caseSummary),1), ...
            'Before',1,'NewVariableNames',{'rho','tol'});
        allSummary = [allSummary;caseSummary]; %#ok<AGROW>

        caseMetrics = metricTable;
        caseMetrics = addvars(caseMetrics, ...
            repmat(rho,height(caseMetrics),1), ...
            repmat(tol,height(caseMetrics),1), ...
            'Before',1,'NewVariableNames',{'rho','tol'});
        allMetrics = [allMetrics;caseMetrics]; %#ok<AGROW>
        writetable(allSummary,fullfile(outputFolder, ...
            'gaussian-n2^14-all-tolerances-summary.csv'));
        writetable(allMetrics,fullfile(outputFolder, ...
            'gaussian-n2^14-all-tolerances-instance-metrics.csv'));
    end
end

function clearGaussianEnvironment()
setenv('LASSO_RHO','');
setenv('LASSO_TOL','');
setenv('LASSO_INSTANCES','');
setenv('LASSO_STOP_CRITERION','');
end
