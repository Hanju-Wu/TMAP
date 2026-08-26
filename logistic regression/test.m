thisDir = fileparts(mfilename('fullpath'));

dataset = 'dataset/news20.binary';
[b,A] = libsvmread(dataset);
[m,n] = size(A);

x0 = zeros(n,1);
mu = 1/m;

%%
opts = struct();
opts.verbose = 0;
opts.maxit = 1000;
opts.ls = 1;
opts.bb = 0;
opts.alpha0 = 1;
opts.gtol = 1e-10;
opts.cg_preconditioner = 'diag';
[x_star, out_star] = lr_2mproj(x0, A, b, mu, opts, 2);


%%
tol = 1e-10;

opts = struct();
opts.verbose = 0;
opts.maxit = 100;
opts.ls = 1;
opts.bb = 0;
opts.alpha0 = 1;
opts.ftol = 1e-11;
opts.gtol = tol;
opts.cg_preconditioner = 'diag';
% Match the 100 correction pairs used by PSSgb and TMP.
opts.lbfgs_mem = 20;
[x_tmap_lbfgs, out_tmap_lbfgs] = lr_2mproj(x0, A, b, mu, opts, 1);
[x_tmap_cg, out_tmap_cg] = lr_2mproj(x0, A, b, mu, opts, 2);

%%
lambda = ones(n,1); % Penalize the absolute value of each element by the same amount
funObj = @(w)LogisticLoss(w,A,b); % Loss function that L1 regularization is applied to
w_init = x0; % Initial value for iterative optimizer
% LogisticLoss returns a sum over samples.  Scaling by 1/m makes the
% L1General objective identical to TMAP's average-loss formulation.
options = struct();
options.quadraticInit = 1;
options.lossType = 'logistic';
options.objectiveScale = 1/m;
[wLASSO_pssgb, out_pssgb] = L1General2_PSSgb(funObj,w_init,lambda,options);

[wLASSO_tmp, out_tmp] = L1General2_TMP(funObj,w_init,lambda,options);

%%
% Use the high-accuracy TMAP-CG run as the common approximate optimum.
f_star = out_star.fvec(end);
scale = max(1,abs(f_star));
err_tmap_cg = max((out_tmap_cg.fvec - f_star)/scale, eps);
err_tmap_lbfgs = max((out_tmap_lbfgs.fvec - f_star)/scale, eps);
err_pssgb = max((out_pssgb.fvec - f_star)/scale, eps);
err_tmp = max((out_tmp.fvec - f_star)/scale, eps);

% Prevent one exceptionally slow solver from compressing the other curves.
runTimes = [out_tmap_cg.timevec(end), out_tmap_lbfgs.timevec(end), ...
    out_pssgb.timevec(end), out_tmp.timevec(end)];
sortedRunTimes = sort(runTimes,'descend');
timeCutoff = max(sortedRunTimes(2),eps);

%%
figure('Name','Logistic objective error versus running time');
semilogy(out_tmap_cg.timevec,err_tmap_cg,'-','LineWidth',2);
hold on;
semilogy(out_tmap_lbfgs.timevec,err_tmap_lbfgs,'--','LineWidth',2);
semilogy(out_pssgb.timevec,err_pssgb,'-.','LineWidth',2);
semilogy(out_tmp.timevec,err_tmp,':','LineWidth',2);
grid on;
xlabel('Running time (s)');
ylabel('Relative objective error');
legend('TMAP-diagPCG','TMAP-L-BFGS','PSSgb','TMP','Location','best');
xlim([0,timeCutoff]);
title('Logistic regression: objective error versus running time');

figure('Name','Logistic sparsity versus running time');
plot(out_tmap_cg.timevec(2:end),out_tmap_cg.dimension(2:end),'-','LineWidth',2);
hold on;
plot(out_tmap_lbfgs.timevec(2:end),out_tmap_lbfgs.dimension(2:end),'--','LineWidth',2);
plot(out_pssgb.timevec(2:end),out_pssgb.nnzvec(2:end),'-.','LineWidth',2);
plot(out_tmp.timevec(2:end),out_tmp.nnzvec(2:end),':','LineWidth',2);
grid on;
xlabel('Running time (s)');
ylabel('nnz(x)');
legend('TMAP-diagPCG','TMAP-L-BFGS','PSSgb','TMP','Location','best');
xlim([0,timeCutoff]);
title('Logistic regression: sparsity versus running time');
