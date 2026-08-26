thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

rho = 0.05;
% signal size
N       = 2^14;
% number of measurements
M       = floor(N/4);
% number of nonzeros
K       = floor(M*rho);

T=int64(rho*M);

x0 = zeros(N,1);
tol = 1e-6;
maxOuterIter = 100;
numInstances = 5;
cputime = nan(numInstances,4);
method = {'PSSgb-L-BFGS','PSSgb-CG','TMAP-CG','TMAP-L-BFGS'};

for i=1:numInstances
%% data
A = 1/sqrt(2*N)*randn(M,N);

e = 1e-2*randn(M,1); %noise
x = sign(sprandn(N,1,K/N));
b = A*x + e;

tau = 0.1*norm(A'*b,'inf');

%% PSSgb-LBFGS
lambda = 2*tau*ones(N,1); % Penalize the absolute value of each element by the same amount
funObj = @(w)SquaredError(w,A,b); % Loss function that L1 regularization is applied to
w_init = x0; % Initial value for iterative optimizer
options.lossType = 'lasso';
options.maxIter = maxOuterIter;
options.objectiveScale = 0.5;
options.optTol = tol;
options.stop_criterion = 'proxResidual';
options.inner_solver = 'lbfgs';
fprintf('\nComputing LASSO Coefficients...\n');
[wLASSO_pssgb_lbfgs,out_pssgb_lbfgs] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);
cputime(i,1) = out_pssgb_lbfgs.runtime;

%% PSSgb-CG
options.inner_solver = 'cg';
options.CG_tol = 1e-1;
options.CG_maxit = 10;
options.CG_adapt = 1;
fprintf('\nComputing PSSgb-CG LASSO coefficients...\n');
[wLASSO_pssgb_cg,out_pssgb_cg] = L1General2_PSSgb( ...
    funObj,w_init,lambda,options);
cputime(i,2) = out_pssgb_cg.runtime;

%% TMAP

opts = struct();
opts.maxit = maxOuterIter;
opts.tol = tol;
opts.x0 = x0;
opts.crit = 1;
opts.cont = 0;
opts.inner_solver = 'cg';
[x_tmap_cg,out_tmap_cg] = tmap(A,b,N,tau,opts);
cputime(i,3) = out_tmap_cg.runtime;

opts.inner_solver = 'lbfgs';
[x_tmap_lbfgs,out_tmap_lbfgs] = tmap(A,b,N,tau,opts);
cputime(i,4) = out_tmap_lbfgs.runtime;

end

% Record one row per instance and one column per method.
cpuTable = array2table(cputime,'VariableNames',method);
cpuTable.instance = (1:numInstances).';
cpuTable = movevars(cpuTable,'instance','Before',1);
disp(cpuTable);

outputFolder = fullfile(fileparts(mfilename('fullpath')),'output');
if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end
writetable(cpuTable,fullfile(outputFolder, ...
    sprintf('high-prox-rho-%.2f-tol-%.0e-cputime.csv',rho,tol)));

% Compare the per-instance CPU times directly.
figure('Color','w');
boxplot(cputime,'Labels',method);
ylabel('CPU time (s)');
grid on;
%title(sprintf('CPU time, \\rho=%.2f, prox residual tolerance=%.0e',rho,tol));
set(gca,'FontSize',10);
saveas(gcf,fullfile(outputFolder, ...
    sprintf('high-prox-rho-%.2f-tol-%.0e-cputime-boxplot.png',rho,tol)));
