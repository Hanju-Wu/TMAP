thisDir = fileparts(mfilename('fullpath'));

datasetOverride = getenv('LASSO_DATASET');
if isempty(datasetOverride)
    dataset = 'dataset/E2006_test';
    % dataset = 'dataset/abalone7.mat';
    % dataset = 'dataset/housing7.mat';
    % dataset = 'dataset/mpg7.mat';
    % dataset = 'dataset/space_ga9.mat';
    % dataset = 'dataset/bodyfat7.mat';
else
    dataset = datasetOverride;
end
[~,datasetName,datasetExtension] = fileparts(dataset);
if strcmpi(datasetExtension,'.mat')
    datasetContents = load(dataset,'A','b');
    A = datasetContents.A;
    b = datasetContents.b;
    clear datasetContents;
else
    [b,A] = libsvmread(dataset);
end

%%
outputFolder = fullfile(thisDir,'output');
if ~exist(outputFolder,'dir'); mkdir(outputFolder); end
[m,n] = size(A);
x0 = zeros(n,1);
regOverride = getenv('LASSO_REG');
if isempty(regOverride)
    reg = 1e-6;
else
    reg = str2double(regOverride);
    if ~isfinite(reg) || reg <= 0
        error('LASSO_REG must be a positive finite number.');
    end
end
tau = reg*norm(A'*b,'inf');
regularizationTag = sprintf('reg-%0.6e',tau);
tol = 1e-10;

%% High-accuracy reference
% The reference solve is not included in any reported comparison time.
referenceOpts = struct();
referenceOpts.maxit = 10000;
referenceOpts.tol = 1e-4;
referenceOpts.x0 = x0;
referenceOpts.crit = 1;
referenceOpts.cont = 0;
referenceOpts.inner_solver = 'cg';
referenceOpts.stopCriterion = 'proxResidual';
[x_reference,out_reference] = tmap(A,b,n,tau,referenceOpts);
fReference = 0.5*norm(A*x_reference-b)^2 + tau*norm(x_reference,1);

%% PSSgb-LBFGS
lambda = 2*tau*ones(n,1); % Penalize the absolute value of each element by the same amount
funObj = @(w)SquaredError(w,A,b); % Loss function that L1 regularization is applied to
w_init = x0; % Initial value for iterative optimizer
options.lossType = 'lasso';
options.maxIter = 10000;
options.objectiveScale = 0.5;
options.verbose = 0;
options.optTol = tol;
options.stopCriterion = 'relativeFunctionGap';
options.fReference = fReference;
options.inner_solver = 'lbfgs';
fprintf('\nComputing PSSgb-L-BFGS coefficients...\n');
[wLASSO_pssgb_lbfgs,out_pssgb_lbfgs] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);

%% PSSgb-CG
options.inner_solver = 'cg';
options.CG_tol = 1e-1;
options.CG_maxit = 10;
options.CG_adapt = 1;
fprintf('\nComputing PSSgb-CG coefficients...\n');
[wLASSO_pssgb_cg,out_pssgb_cg] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);

%% TMAP-cg

opts = struct();
opts.maxit = 10000;
opts.tol = tol;
opts.x0 = x0;
opts.crit = 1;
opts.cont = 0;
opts.stopCriterion = 'relativeFunctionGap';
opts.fReference = fReference;
opts.inner_solver = 'cg';
[x_tmap_cg,out_tmap_cg] = tmap(A,b,n,tau,opts);

%% TMAP-LBFGS
opts.inner_solver = 'lbfgs';
[x_tmap_lbfgs,out_tmap_lbfgs] = tmap(A,b,n,tau,opts);

%% result
f_star = fReference;
scale = max(1,abs(f_star));
err_tmap_cg = max((out_tmap_cg.fvec - f_star)/scale,eps);
err_tmap_lbfgs = max((out_tmap_lbfgs.fvec - f_star)/scale,eps);
err_pssgb_cg = max((out_pssgb_cg.fvec - f_star)/scale,eps);
err_pssgb_lbfgs = max((out_pssgb_lbfgs.fvec - f_star)/scale,eps);

runTimes = [out_tmap_cg.timevec(end), out_tmap_lbfgs.timevec(end), ...
    out_pssgb_cg.timevec(end),out_pssgb_lbfgs.timevec(end)];
sortedRunTimes = sort(runTimes,'descend');

timeCutoff = sortedRunTimes(1);

method = {'TMAP-CG';'TMAP-L-BFGS';'PSSgb-CG';'PSSgb-L-BFGS'};
runtime = runTimes(:);
finalObjective = [out_tmap_cg.fvec(end);out_tmap_lbfgs.fvec(end); ...
    out_pssgb_cg.fvec(end);out_pssgb_lbfgs.fvec(end)];
relativeObjectiveError = [err_tmap_cg(end);err_tmap_lbfgs(end); ...
    err_pssgb_cg(end);err_pssgb_lbfgs(end)];
solutionNnz = [out_tmap_cg.nnzvec(end);out_tmap_lbfgs.nnzvec(end); ...
    out_pssgb_cg.nnzvec(end);out_pssgb_lbfgs.nnzvec(end)];
matrixProducts = [out_tmap_cg.mvp_total;out_tmap_lbfgs.mvp_total; ...
    out_pssgb_cg.mvp_total;out_pssgb_lbfgs.mvp_total];
comparisonTable = table(method,runtime,finalObjective, ...
    relativeObjectiveError,solutionNnz,matrixProducts);
disp(comparisonTable);

%% picture
plotColors = lines(4);
plotLineWidth = 2.5;
figObjective = figure('Name','LASSO objective error versus running time');
semilogy(out_tmap_cg.timevec,err_tmap_cg,'-', ...
    'Color',plotColors(1,:),'LineWidth',plotLineWidth);
hold on;
semilogy(out_tmap_lbfgs.timevec,err_tmap_lbfgs,'--', ...
    'Color',plotColors(2,:),'LineWidth',plotLineWidth);
semilogy(out_pssgb_cg.timevec,err_pssgb_cg,'-.', ...
    'Color',plotColors(3,:),'LineWidth',plotLineWidth);
semilogy(out_pssgb_lbfgs.timevec,err_pssgb_lbfgs,':', ...
    'Color',plotColors(4,:),'LineWidth',plotLineWidth);
grid on;
xlabel('Running time (s)');
ylabel('Relative objective error');
legend('TMAP-CG','TMAP-L-BFGS','PSSgb-CG','PSSgb-L-BFGS', ...
    'Location','best');
xlim([0,timeCutoff]);
exportgraphics(figObjective, ...
    fullfile(outputFolder,[datasetName '-' regularizationTag '-relative-error.png']), ...
    'Resolution',300);

figNnz = figure('Name','LASSO sparsity versus running time');
plot(out_tmap_cg.timevec(2:end),out_tmap_cg.nnzvec(2:end), ...
    '-','Color',plotColors(1,:),'LineWidth',plotLineWidth);
hold on;
plot(out_tmap_lbfgs.timevec(2:end),out_tmap_lbfgs.nnzvec(2:end), ...
    '--','Color',plotColors(2,:),'LineWidth',plotLineWidth);
plot(out_pssgb_cg.timevec(2:end),out_pssgb_cg.nnzvec(2:end), ...
    '-.','Color',plotColors(3,:),'LineWidth',plotLineWidth);
plot(out_pssgb_lbfgs.timevec(2:end),out_pssgb_lbfgs.nnzvec(2:end), ...
    ':','Color',plotColors(4,:),'LineWidth',plotLineWidth);
grid on;
xlabel('Running time (s)');
ylabel('nnz(x)');
legend('TMAP-CG','TMAP-L-BFGS','PSSgb-CG','PSSgb-L-BFGS', ...
    'Location','best');
xlim([0,timeCutoff]);
% exportgraphics(figNnz, ...
%     fullfile(outputFolder,[datasetName '-' regularizationTag '-nnz-time.png']), ...
%     'Resolution',300);
