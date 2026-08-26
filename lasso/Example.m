thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

rhoOverride = getenv('LASSO_RHO');
if isempty(rhoOverride)
    rho = 0.1;
else
    rho = str2double(rhoOverride);
end

% signal size
N       = 2^14;
% number of measurements
M       = floor(N/4);
% number of nonzeros
K       = floor(M*rho);

T=int64(rho*M);
tolOverride = getenv('LASSO_TOL');
if isempty(tolOverride)
    tol = 1e-6;
else
    tol = str2double(tolOverride);
end
% Number of independently generated instances.  Set this to 100 for the
% paper-scale experiment; a smaller value is useful for a quick smoke test.
instanceOverride = getenv('LASSO_INSTANCES');
if isempty(instanceOverride)
    numInstances = 10;
else
    numInstances = str2double(instanceOverride);
end
criterionOverride = getenv('LASSO_STOP_CRITERION');
if isempty(criterionOverride)
    stopCriterion = 'proxResidual';
else
    stopCriterion = criterionOverride;
end
maxOuterIter = 1000;
times = nan(numInstances,7);
rawMvp = nan(numInstances,7);
outerIterations = nan(numInstances,7);
proxResidual = nan(numInstances,7);
relativeRecoveryError = nan(numInstances,7);
relativeFunctionGap = nan(numInstances,7);
success = false(numInstances,7);

x0 = zeros(N,1);

for i=1:numInstances
%% data
A = 1/sqrt(2*N)*randn(M,N);

e = 1e-2*randn(M,1); %noise
x = sign(sprandn(N,1,K/N));
b = A*x + e;

tau = 0.1*norm(A'*b,'inf');

% Build one high-accuracy reference objective for this instance. The
% reference solve is excluded from all reported runtimes.
refOpts = struct('maxit',100,'tol',1e-8,'x0',x0,'crit',1, ...
    'cont',1,'inner_solver','cg','stopCriterion','proxResidual');
[xRef,~] = tmap(A,b,N,tau,refOpts);
fReference = lassoObjective(xRef,A,b,tau);

%% PSSgb-LBFGS
lambda = 2*tau*ones(N,1); % Penalize the absolute value of each element by the same amount
funObj = @(w)SquaredError(w,A,b); % Loss function that L1 regularization is applied to
w_init = x0; % Initial value for iterative optimizer
options.lossType = 'lasso';
options.maxIter = maxOuterIter;
options.objectiveScale = 0.5;
options.optTol = tol;
options.stop_criterion = 'proxResidual';
options.stopCriterion = stopCriterion;
options.fReference = fReference;
options.inner_solver = 'lbfgs';
fprintf('\nComputing LASSO Coefficients...\n');
[wLASSO_pssgb_lbfgs,out_pssgb_lbfgs] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);

%% PSSgb-CG
options.inner_solver = 'cg';
options.CG_tol = 1e-1;
options.CG_maxit = 10;
options.CG_adapt = 1;
fprintf('\nComputing PSSgb-CG LASSO coefficients...\n');
[wLASSO_pssgb_cg,out_pssgb_cg] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);

%% FAST-2CDA_ehn
verbosity=0;
delta=0.8;

tic;

[xour2acc,tottime2acc,err2acc,errlsq2acc,...
    iter2acc, f2acc, errVec2acc, timeVec2acc,res2acc,fast_mvp] =  ...
    Call_FAST_2CDA_ehn(A,b, x, tau,...
    tol,int64(0.8*delta*T),'stopCriterion',stopCriterion,'fref',fReference);

t2acc = toc;

%% SpaRSA

opts = struct();
    opts.maxit = maxOuterIter;
opts.c = 1e-2;
opts.M = 5;
opts.ssu = 1e+30;
opts.ssl = 1e-30;
opts.beta = 0.2;   
opts.tol = tol;
opts.stopCriterion = stopCriterion;
opts.fReference = fReference;
[funv1, x1, resi1, numit1, cput1, ~, spaRSA_mvp] = ...
    alg_nmt(A, b, x0, tau, opts);

%% TMAP

opts = struct();
opts.maxit = maxOuterIter;
opts.tol = tol;
opts.x0 = x0;
opts.crit = 1;
opts.cont = 0;
opts.stopCriterion = stopCriterion;
opts.fReference = fReference;
opts.inner_solver = 'cg';
[x_tmap_cg,out_tmap_cg] = tmap(A,b,N,tau,opts);

opts.inner_solver = 'lbfgs';
[x_tmap_lbfgs,out_tmap_lbfgs] = tmap(A,b,N,tau,opts);

%% ASSN
opts = struct();
opts.maxit = maxOuterIter;
opts.tol = tol;
opts.x0 = x0;
opts.crit = 1;
if strcmpi(stopCriterion,'relativeFunctionGap')
    opts.crit = 2;
    opts.fopt = fReference;
end
opts.cont = 0;
[x_ssn,out_ssn]     = ssmNewtonL1Pen(A,b,N,tau,opts);
%% Unified per-instance metrics
solutions = {wLASSO_pssgb_lbfgs,wLASSO_pssgb_cg,xour2acc,x1,...
    x_tmap_cg,x_tmap_lbfgs,x_ssn};
times(i,:) = [out_pssgb_lbfgs.runtime out_pssgb_cg.runtime ...
    t2acc cput1 out_tmap_cg.runtime ...
    out_tmap_lbfgs.runtime out_ssn.time];
rawMvp(i,:) = [out_pssgb_lbfgs.mvp_total(end) out_pssgb_cg.mvp_total(end) ...
    fast_mvp(end) spaRSA_mvp(end) out_tmap_cg.mvp_total(end) ...
    out_tmap_lbfgs.mvp_total(end) out_ssn.Acalls(end)];
outerIterations(i,:) = [out_pssgb_lbfgs.iter out_pssgb_cg.iter ...
    iter2acc numit1 out_tmap_cg.iter ...
    out_tmap_lbfgs.iter out_ssn.iter];
for methodIndex = 1:7
    [proxResidual(i,methodIndex),relativeRecoveryError(i,methodIndex), ...
        relativeFunctionGap(i,methodIndex)] = ...
        lassoInstanceMetrics(solutions{methodIndex},A,b,tau,x,fReference);
    if strcmpi(stopCriterion,'relativeFunctionGap')
        success(i,methodIndex) = isfinite(relativeFunctionGap(i,methodIndex)) && ...
            relativeFunctionGap(i,methodIndex) <= tol;
    else
        success(i,methodIndex) = isfinite(proxResidual(i,methodIndex)) && ...
            proxResidual(i,methodIndex) <= tol;
    end
end
fprintf('Instance %d/%d complete. Success rate so far: %s\n',i,numInstances,...
    sprintf('%.1f%% ',100*mean(success(1:i,:),1)));
end

method = {'PSSgb-L-BFGS';'PSSgb-CG';'FAST-2CDA-E'; ...
    'SpaRSA';'TMAP-CG';'TMAP-L-BFGS';'ASSN'};
figure;
plotData = times;
plotData(~success) = NaN;
validGroups = any(isfinite(plotData),1);
if size(plotData,1) >= 2 && any(validGroups)
    boxplot(plotData(:,validGroups), 'Labels', method(validGroups).');
elseif size(plotData,1) == 1
    plot(times, 'o');
    set(gca,'XTick',1:numel(method),'XTickLabel',method);
    title('One-instance timing (boxplot requires at least two observations)');
else
    close(gcf);
end
ylabel('CPU time (s)');
grid on;

successfulTimes = times;
successfulTimes(~success) = NaN;
successfulMvp = rawMvp;
successfulMvp(~success) = NaN;
comparisonTable = table(method,mean(successfulTimes,1,'omitnan').', ...
    std(successfulTimes,0,1,'omitnan').',mean(successfulMvp,1,'omitnan').', ...
    std(successfulMvp,0,1,'omitnan').',mean(success,1).', ...
    'VariableNames',{'method','meanTime','stdTime','meanRawMVP', ...
    'stdRawMVP','successRate'});
disp(comparisonTable);

metricTable = table();
for methodIndex = 1:7
    rows = table(repmat({method{methodIndex}},numInstances,1), ...
        (1:numInstances).',times(:,methodIndex),rawMvp(:,methodIndex), ...
        outerIterations(:,methodIndex),proxResidual(:,methodIndex), ...
        relativeRecoveryError(:,methodIndex),relativeFunctionGap(:,methodIndex), ...
        success(:,methodIndex), ...
        'VariableNames',{'method','instance','cpuTime','rawMVP', ...
        'outerIterations','proxResidual','relativeRecoveryError', ...
        'relativeFunctionGap','success'});
    metricTable = [metricTable;rows]; %#ok<AGROW>
end
outputFolder = fullfile(thisDir,'output');
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
writetable(metricTable,fullfile(outputFolder, ...
    sprintf('gaussian-n2^14-rho-%.2f-tol-%.0e-instance-metrics.csv',rho,tol)));

summaryTable = table(method,mean(successfulTimes,1,'omitnan').', ...
    std(successfulTimes,0,1,'omitnan').',mean(successfulMvp,1,'omitnan').', ...
    std(successfulMvp,0,1,'omitnan').',mean(outerIterations,1,'omitnan').', ...
    mean(proxResidual,1,'omitnan').',mean(relativeRecoveryError,1,'omitnan').', ...
    mean(relativeFunctionGap,1,'omitnan').', ...
    mean(success,1).', ...
    'VariableNames',{'method','meanTime','stdTime','meanRawMVP','stdRawMVP', ...
    'meanOuterIterations','meanProxResidual','meanRelativeRecoveryError', ...
    'meanRelativeFunctionGap', ...
    'successRate'});
disp(summaryTable);
writetable(summaryTable,fullfile(outputFolder, ...
    sprintf('gaussian-n2^14-rho-%.2f-tol-%.0e-summary.csv',rho,tol)));

function [residual,recoveryError,functionGap] = lassoInstanceMetrics(w,A,b,tau,xTrue,fReference)
% Use the same unit-step proximal residual for every solver.
if isempty(w) || any(~isfinite(w))
    residual = Inf;
    recoveryError = Inf;
    functionGap = Inf;
    return;
end
gradient = A'*(A*w-b);
proxPoint = sign(w-gradient).*max(abs(w-gradient)-tau,0);
residual = norm(w-proxPoint);
recoveryError = norm(w-xTrue)/max(norm(xTrue),eps);
functionValue = lassoObjective(w,A,b,tau);
functionGap = (functionValue-fReference)/max(1,abs(fReference));
end

function value = lassoObjective(w,A,b,tau)
value = 0.5*norm(A*w-b)^2 + tau*norm(w,1);
end
