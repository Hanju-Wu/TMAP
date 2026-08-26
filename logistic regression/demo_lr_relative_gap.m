% Plot relative function error versus running time for logistic regression.
% A TMAP-CG run with prox-residual tolerance 1e-6 supplies the reference.
clc; clear;
addpath('IRPN'); addpath('dataset');
addpath(genpath(fullfile('..','lasso','L1GeneralExamples')));

datasets = {'rcv1_train.binary','rcv1_test.binary','news20.binary', ...
    'real-sim','a9a','ijcnn1'};
outdir = fullfile('..','..','Springer-Nature-template','output');
if ~exist(outdir,'dir'); mkdir(outdir); end

for id = 1:numel(datasets)
    fprintf('\n===== %s =====\n',datasets{id});
    [b,A] = libsvmread(fullfile('dataset',datasets{id}));
    [m,n] = size(A); x0 = zeros(n,1); mu = 1/m;

    % Stage 1: reference point from TMAP-CG at prox-residual 1e-6.
    refOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',0, ...
        'alpha0',1,'ftol',0,'gtol',1e-6,'cg_preconditioner','diag');
    [~,refOut] = lr_2mproj(x0,A,b,mu,refOpts,2);
    f_ref = refOut.fvec(end);

    % Stage 2: compare all methods using relative function error.
    gapTol = 1e-10;
    cmpOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',0,'alpha0',1, ...
        'ftol',0,'gtol',1e-12,'f_ref',f_ref,'gap_tol',gapTol, ...
        'cg_preconditioner','diag','lbfgs_mem',20);
    [~,out_tmap_cg] = lr_2mproj(x0,A,b,mu,cmpOpts,2);
    [~,out_tmap_lbfgs] = lr_2mproj(x0,A,b,mu,cmpOpts,1);

    funObj = @(w)LogisticLoss(w,A,b);
    lambda = ones(n,1);
    pssgbOpts = struct('verbose',0,'maxIter',1000,'optTol',gapTol, ...
        'lossType','logistic','objectiveScale',1/m,'quadraticInit',1, ...
        'stopCriterion','relativeFunctionGap','fReference',f_ref, ...
        'mvpPerEval',[1,1]);
    pssgbOpts.innerSolver = 'cg';
    [~,out_pssgb_cg] = L1General2_PSSgb(funObj,x0,lambda,pssgbOpts);
    pssgbOpts.innerSolver = 'lbfgs';
    [~,out_pssgb_lbfgs] = L1General2_PSSgb(funObj,x0,lambda,pssgbOpts);

    altOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',1,'alpha0',1, ...
        'ftol',0,'gtol',1e-12,'f_ref',f_ref,'gap_tol',gapTol);
    [~,out_altn] = lr_newtonacc(x0,A,b,mu,altOpts);

    data.A = A; data.b = b;
    model.loss = 'logistic'; model.penalty = 'ell1';
    model.regpara = mu; model.eps = 1e-12;
    model.f_ref = f_ref; model.gap_tol = gapTol;
    irpnOpts = struct('maxit',100,'maxitsub',100,'mu',1e-6,'eta',0.5, ...
        'rho',0.5,'beta',0.25,'sigma',0.25);
    [~,~,~,~,~,~,~,stats_irpn] = alg_rpn(x0,data,model,irpnOpts);

    stats = {
        makeStats(out_tmap_cg.fvec,out_tmap_cg.timevec,out_tmap_cg.mvp,f_ref), ...
        makeStats(out_tmap_lbfgs.fvec,out_tmap_lbfgs.timevec,out_tmap_lbfgs.mvp,f_ref), ...
        makeStats(out_pssgb_cg.fvec,out_pssgb_cg.timevec,out_pssgb_cg.mvp,f_ref), ...
        makeStats(out_pssgb_lbfgs.fvec,out_pssgb_lbfgs.timevec,out_pssgb_lbfgs.mvp,f_ref), ...
        makeStats(out_altn.fvec,out_altn.timevec(2:end),out_altn.mvpvec(2:end),f_ref), ...
        stats_irpn};
    labels = {'a','b','c','d','e','f'};
    styles = {'-','--','-.',':','-','--'};

    figure('Visible','off'); hold on;
    for j = 1:numel(stats)
        semilogy(stats{j}.timevec,max(stats{j}.gapvec,1e-16),styles{j},'LineWidth',1.8);
    end
    grid on; set(gca,'YScale','log'); ylim([1e-16,1]);
    xlabel('Running time (s)'); ylabel('Relative function error');
    legend(labels,'Location','best');
    % title(['Logistic regression: ',datasets{id},' (a-f defined in Table 2)']);
    name = erase(datasets{id},'.binary');
    saveas(gcf,fullfile(outdir,[name,'-relative-error-vs-time.png']));
    close(gcf);
end

function stats = makeStats(fvec,timevec,mvpvec,f_ref)
stats.fvec = fvec(:);
stats.timevec = timevec(:);
stats.mvpvec = mvpvec(:);
stats.gapvec = relative_objective_gap(stats.fvec,f_ref);
end
