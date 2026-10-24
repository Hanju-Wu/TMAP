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
T = int64(rho*M);
delta = 0.8;
fastMemory = int64(0.8*delta*T); % maximum candidate coordinates per FAST sweep

tolerancesOverride = getenv('LASSO_TOLS');
tolOverride = getenv('LASSO_TOL'); % legacy single-target entry point
if ~isempty(tolerancesOverride)
    tolerances = sscanf(tolerancesOverride,'%f');
elseif ~isempty(tolOverride)
    tolerances = str2double(tolOverride);
else
    tolerances = [1e-4;1e-8;1e-12];
end
validateattributes(tolerances,{'numeric'}, ...
    {'vector','real','positive','finite','nonempty'});
tolerances = unique(tolerances(:),'stable');
tol = min(tolerances); % one solver call, continuing to the strictest target
% Number of independently generated instances: 10 per sparsity level.
% A smaller value is useful for a quick smoke test.
instanceOverride = getenv('LASSO_INSTANCES');
if isempty(instanceOverride)
    numInstances = 10;
else
    numInstances = str2double(instanceOverride);
end
criterionOverride = getenv('LASSO_STOP_CRITERION');
if isempty(criterionOverride)
    stopCriterion = 'relativeFunctionGap';
else
    stopCriterion = criterionOverride;
end
if ~strcmpi(stopCriterion,'relativeFunctionGap')
    error('Gaussian:StopCriterion', ...
        'Gaussian checkpoint tables require relativeFunctionGap stopping.');
end
maxOuterIter = 1000;
method = {'PSSgb-L-BFGS';'FAST-2CDA-E'; ...
    'SpaRSA';'TMAP-CG';'TMAP-L-BFGS';'ASSN'};
runHistory = cell(numInstances,numel(method));
references = cell(numInstances,1);
metricTable = table();
outputFolder = fullfile(thisDir,'output');
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
outputStem = sprintf('gaussian-n2^%d-rho-%.2f',round(log2(N)),rho);

x0 = zeros(N,1);

for i=1:numInstances
%% data
% Fixed seed per (rho, instance) so every instance can be regenerated.
instanceSeed = 1000*round(100*rho) + i;
rng(instanceSeed,'twister');
instanceRng = rng;
A = 1/sqrt(2*N)*randn(M,N);

e = 1e-2*randn(M,1); %noise
x = sign(sprandn(N,1,K/N));
b = A*x + e;

tau = 0.1*norm(A'*b,'inf');

% Build one high-accuracy reference objective for this instance. The
% reference solve is excluded from all reported runtimes.
refOpts = struct('maxit',1000,'tol',1e-10,'x0',x0,'crit',1, ...
    'cont',1,'inner_solver','cg','stopCriterion','proxResidual', ...
    'allowForcedAcceptance',false);
[xRef,refOut] = tmap(A,b,N,tau,refOpts);
refGradient = A'*(A*xRef-b);
refProxPoint = sign(xRef-refGradient).*max(abs(xRef-refGradient)-tau,0);
referenceResidual = norm(xRef-refProxPoint);
if ~isfinite(referenceResidual) || referenceResidual > refOpts.tol
    error('Gaussian:ReferenceNotConverged', ...
        ['Instance %d: TMAP-CG reference residual %.3e exceeds %.3e ', ...
        'after %d iterations. Comparison aborted.'], ...
        i,referenceResidual,refOpts.tol,refOut.iter);
end
fReference = lassoObjective(xRef,A,b,tau);
fprintf('Reference TMAP-CG: residual %.3e, %d iterations, f_ref %.16e\n', ...
    referenceResidual,refOut.iter,fReference);
references{i} = struct('fReference',fReference,'residual',referenceResidual, ...
    'tol',refOpts.tol,'iter',refOut.iter,'runtime',refOut.runtime, ...
    'seed',instanceSeed,'rng',instanceRng);

%% PSSgb-LBFGS
lambda = 2*tau*ones(N,1); % Penalize the absolute value of each element by the same amount
funObj = @(w)SquaredError(w,A,b); % Loss function that L1 regularization is applied to
w_init = x0; % Initial value for iterative optimizer
options = struct();
options.lossType = 'lasso';
options.maxIter = maxOuterIter;
options.objectiveScale = 0.5;
options.optTol = tol;
options.stopCriterion = stopCriterion;
options.fReference = fReference;
options.inner_solver = 'lbfgs';
fprintf('\nComputing LASSO Coefficients...\n');
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);
options.recordFcn = recordPoint;
[wLASSO_pssgb_lbfgs,out_pssgb_lbfgs] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);
runHistory{i,1} = finishRecord();

%% FAST-2CDA_ehn
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);

[xour2acc,tottime2acc,err2acc,errlsq2acc,...
    iter2acc, f2acc, errVec2acc, timeVec2acc,res2acc,fast_mvp] =  ...
    Call_FAST_2CDA_ehn(A,b, x, tau,...
    tol,'memory',fastMemory,'stopCriterion',stopCriterion,'fref',fReference, ...
    'recordFcn',recordPoint);
runHistory{i,2} = finishRecord();

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
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);
opts.recordFcn = recordPoint;
[funv1, x1, resi1, numit1, cput1, ~, spaRSA_mvp] = ...
    alg_nmt(A, b, x0, tau, opts);
runHistory{i,3} = finishRecord();

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
opts.allowForcedAcceptance = false;
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);
opts.recordFcn = recordPoint;
[x_tmap_cg,out_tmap_cg] = tmap(A,b,N,tau,opts);
runHistory{i,4} = finishRecord();

opts.inner_solver = 'lbfgs';
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);
opts.recordFcn = recordPoint;
[x_tmap_lbfgs,out_tmap_lbfgs] = tmap(A,b,N,tau,opts);
runHistory{i,5} = finishRecord();

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
[recordPoint,finishRecord] = lasso_gap_recorder(fReference,tolerances);
opts.recordFcn = recordPoint;
[x_ssn,out_ssn] = ssmNewtonL1Pen(A,b,N,tau,opts);
runHistory{i,6} = finishRecord();
%% First crossings from the six single-run trajectories
for methodIndex = 1:numel(method)
    rows = lasso_gap_checkpoint_metrics(runHistory{i,methodIndex},A,b,tau,x);
    rows = addvars(rows,repmat(method(methodIndex),height(rows),1), ...
        repmat(i,height(rows),1),'Before',1, ...
        'NewVariableNames',{'method','instance'});
    rows.fReference = repmat(fReference,height(rows),1);
    rows.referenceResidual = repmat(referenceResidual,height(rows),1);
    metricTable = [metricTable;rows]; %#ok<AGROW>
end
fprintf('Instance %d/%d complete; all gaps extracted from the same runs.\n', ...
    i,numInstances);
% Persist completed trajectories and first-crossing snapshots after each
% instance; data generation and reference solves are not repeated per target.
save(fullfile(outputFolder,[outputStem,'-trajectories.mat']), ...
    'runHistory','references','metricTable','tolerances','method','rho', ...
    'N','M','numInstances','maxOuterIter','fastMemory','stopCriterion');
end

summaryTable = summarizeCheckpoints(metricTable,method,tolerances);
disp(summaryTable);
for q = 1:numel(tolerances)
    target = tolerances(q);
    targetMetrics = metricTable(metricTable.tol == target,:);
    targetSummary = summaryTable(summaryTable.tol == target,:);
    writetable(targetMetrics,fullfile(outputFolder, ...
        sprintf('%s-tol-%.0e-instance-metrics.csv',outputStem,target)));
    writetable(targetSummary,fullfile(outputFolder, ...
        sprintf('%s-tol-%.0e-summary.csv',outputStem,target)));
    plotCheckpointTimes(targetMetrics,method,numInstances,target);
end

function summary = summarizeCheckpoints(metrics,method,tolerances)
summary = table();
for q = 1:numel(tolerances)
    for j = 1:numel(method)
        d = metrics(metrics.tol == tolerances(q) & strcmp(metrics.method,method{j}),:);
        row = table(tolerances(q),method(j), ...
            mean(d.cpuTime,'omitnan'),std(d.cpuTime,0,'omitnan'), ...
            mean(d.rawMVP,'omitnan'),std(d.rawMVP,0,'omitnan'), ...
            mean(d.outerIterations,'omitnan'),mean(d.proxResidual,'omitnan'), ...
            mean(d.relativeRecoveryError,'omitnan'), ...
            mean(d.relativeFunctionGap,'omitnan'),mean(d.success), ...
            'VariableNames',{'tol','method','meanTime','stdTime','meanRawMVP', ...
            'stdRawMVP','meanOuterIterations','meanProxResidual', ...
            'meanRelativeRecoveryError','meanRelativeFunctionGap','successRate'});
        summary = [summary;row]; %#ok<AGROW>
    end
end
end

function plotCheckpointTimes(metrics,method,numInstances,target)
times = NaN(numInstances,numel(method));
for j = 1:numel(method)
    d = metrics(strcmp(metrics.method,method{j}),:);
    times(d.instance,j) = d.cpuTime;
end
validGroups = any(isfinite(times),1);
if ~any(validGroups); return; end
figure;
if numInstances >= 2
    boxplot(times(:,validGroups),'Labels',method(validGroups).');
else
    plot(times,'o');
    set(gca,'XTick',1:numel(method),'XTickLabel',method);
end
ylabel('Elapsed time (s)');
title(sprintf('First relative gap <= %.0e',target));
grid on;
end

function value = lassoObjective(w,A,b,tau)
value = 0.5*norm(A*w-b)^2 + tau*norm(w,1);
end
