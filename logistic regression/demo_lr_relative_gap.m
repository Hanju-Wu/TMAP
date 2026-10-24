% Plot relative function error versus running time for logistic regression.
% A TMAP-CG run with prox-residual tolerance 1e-10 supplies the reference.
clc;
lrDir = fileparts(mfilename('fullpath'));
lassoDir = fullfile(fileparts(lrDir),'lasso');
addpath(lrDir,fullfile(lrDir,'IRPN'),fullfile(lrDir,'dataset'),lassoDir);
addpath(genpath(fullfile(lassoDir,'L1GeneralExamples')));

datasets = {'rcv1_train.binary','rcv1_test.binary','news20.binary', ...
    'real-sim','a9a','ijcnn1'};
limitedDatasets = {'rcv1_test.binary','news20.binary'};
largeDatasetMaxTime = [200,25]; % seconds per solver/reference, in limitedDatasets order
% Reference and comparison gap targets are separate.
tol = 1e-10;
outdir = fullfile(lrDir,'output');
if ~exist(outdir,'dir'); mkdir(outdir); end

for id = 1:numel(datasets)
    fprintf('\n===== %s =====\n',datasets{id});
    [b,A] = libsvmread(fullfile(lrDir,'dataset',datasets{id}));
    [m,n] = size(A); x0 = zeros(n,1); mu = 1/m;
    maxTime = Inf;
    [isLimited,limitIndex] = ismember(datasets{id},limitedDatasets);
    if isLimited
        maxTime = largeDatasetMaxTime(limitIndex);
    end

    tmapOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',0,'alpha0',1, ...
        'ftol',0,'gtol',tol,'cg_preconditioner','none','cg_adaptive',true, ...
        'cg_unlimited_factor',100,'lbfgs_mem',100,'maxTime',maxTime);

    % Stage 1: reference point from TMAP-CG at prox-residual 1e-10.
    refOpts = tmapOpts;
    refOpts.gtol = 1e-10;
    [~,refOut] = lr_2mproj(x0,A,b,mu,refOpts,2);
    f_ref = refOut.fvec(end);
    referenceConverged = refOut.res(end) <= refOpts.gtol;
    reportRun('reference TMAP-CG',refOut);
    if ~referenceConverged
        error('demo_lr_relative_gap:IncompleteReference', ...
            '%s reference residual %.3e exceeds %.3e; comparison aborted.', ...
            datasets{id},refOut.res(end),refOpts.gtol);
    end

    % Stage 2: compare all methods using relative function error.
    gapTol = 1e-15;
    cmpOpts = tmapOpts;
    cmpOpts.f_ref = f_ref;
    cmpOpts.gap_tol = gapTol;
    [~,out_tmap_cg] = lr_2mproj(x0,A,b,mu,cmpOpts,2);
    reportRun('a: TMAP-CG',out_tmap_cg);
    [~,out_tmap_lbfgs] = lr_2mproj(x0,A,b,mu,cmpOpts,1);
    reportRun('b: TMAP-LBFGS',out_tmap_lbfgs);

    funObj = @(w)LogisticLoss(w,A,b);
    lambda = ones(n,1);
    pssgbOpts = struct('verbose',0,'maxIter',1000,'optTol',gapTol, ...
        'lossType','logistic','objectiveScale',1/m,'quadraticInit',1, ...
        'stopCriterion','relativeFunctionGap','fReference',f_ref, ...
        'mvpPerEval',[1,1],'cgTol',1e-1,'cgMaxIter',10,'cgAdapt',1, ...
        'corrections',100,'maxTime',maxTime);
    pssgbOpts.innerSolver = 'lbfgs';
    [~,out_pssgb_lbfgs] = L1General2_PSSgb(funObj,x0,lambda,pssgbOpts);
    reportRun('c: PSSgb-LBFGS',out_pssgb_lbfgs);

    altOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',1,'alpha0',1, ...
        'ftol',0,'gtol',tol,'f_ref',f_ref,'gap_tol',gapTol,'maxTime',maxTime);
    [~,out_altn] = lr_newtonacc(x0,A,b,mu,altOpts);
    reportRun('d: acc Newton',out_altn);

    data.A = A; data.b = b;
    model.loss = 'logistic'; model.penalty = 'ell1';
    model.regpara = mu; model.eps = tol;
    model.f_ref = f_ref; model.gap_tol = gapTol;
    irpnOpts = struct('maxit',100,'maxitsub',100,'mu',1e-6,'eta',0.5, ...
        'rho',0.5,'beta',0.25,'sigma',0.25,'maxTime',maxTime);
    [~,~,~,~,~,~,~,stats_irpn] = alg_rpn(x0,data,model,irpnOpts);
    reportRun('e: IRPN',stats_irpn);

    stats = {
        makeStats(out_tmap_cg.fvec,out_tmap_cg.timevec,out_tmap_cg.mvp,f_ref,out_tmap_cg), ...
        makeStats(out_tmap_lbfgs.fvec,out_tmap_lbfgs.timevec,out_tmap_lbfgs.mvp,f_ref,out_tmap_lbfgs), ...
        makeStats(out_pssgb_lbfgs.fvec,out_pssgb_lbfgs.timevec,out_pssgb_lbfgs.mvp,f_ref,out_pssgb_lbfgs), ...
        makeStats([refOut.fvec(1);out_altn.fvec(:)],out_altn.timevec,out_altn.mvpvec,f_ref,out_altn), ...
        stats_irpn};
    labels = {'TMAP-CG','TMAP-L-BFGS','PSSgb-L-BFGS','AltN','IRPN'};
    styles = {'-','--','-.',':','-'};
    name = erase(datasets{id},'.binary');
    % Keep the measured partial trajectories and reference quality alongside the plot.
    save(fullfile(outdir,[name,'-relative-gap.mat']), ...
        'stats','labels','f_ref','refOut','referenceConverged','maxTime','gapTol', ...
        'refOpts','cmpOpts','pssgbOpts','altOpts','irpnOpts');

    figure('Visible','off'); hold on;
    plotLabels = labels;
    for j = 1:numel(stats)
        keep = stats{j}.timevec <= maxTime;
        curve = semilogy(stats{j}.timevec(keep),max(stats{j}.gapvec(keep),1e-16),styles{j},'LineWidth',1.8);
        if nnz(keep) == 1; set(curve,'Marker','o'); end
        if stats{j}.timedOut
            plotLabels{j} = sprintf('%s (time limit)',labels{j});
        end
    end
    grid on; set(gca,'YScale','log'); ylim([1e-16,1]);
    if isfinite(maxTime); xlim([0,maxTime]); end
    xlabel('Running time (s)'); ylabel('Relative function error');
    legend(plotLabels,'Location','best');
    if ~referenceConverged
        title(sprintf('Approximate reference: residual %.2e',refOut.res(end)));
    end
    
    saveas(gcf,fullfile(outdir,[name,'-relative-error-vs-time.png']));
    close(gcf);
end

function stats = makeStats(fvec,timevec,mvpvec,f_ref,out)
stats.fvec = fvec(:);
stats.timevec = timevec(:);
stats.mvpvec = mvpvec(:);
stats.gapvec = relative_objective_gap(stats.fvec,f_ref);
stats.timedOut = out.timedOut;
stats.runtime = out.runtime;
stats.maxTime = out.maxTime;
end

function reportRun(label,out)
if out.timedOut
    fprintf('%s: %.2f s, time limit reached; retained the last accepted iterate.\n', ...
        label,out.runtime);
else
    fprintf('%s: %.2f s.\n',label,out.runtime);
end
end
