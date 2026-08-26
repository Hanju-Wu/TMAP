% Logistic-regression comparison on all LIBSVM data sets.
% Reports one table per data set with CPU time and matrix-vector products N_A.
clc; clear;
addpath('IRPN'); addpath('dataset');
addpath(genpath(fullfile('..','lasso','L1GeneralExamples')));

datasets = {'rcv1_train.binary','rcv1_test.binary','news20.binary', ...
    'real-sim','a9a','ijcnn1'};
methodNames = {'a';'b';'c';'d';'e';'f'};
tol = 1e-6;

for id = 1:numel(datasets)
    dataset = fullfile('dataset',datasets{id});
    fprintf('\n===== %s =====\n',datasets{id});
    [b,A] = libsvmread(dataset);
    [m,n] = size(A);
    x0 = zeros(n,1);
    mu = 1/m;

    tmapOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',0,'alpha0',1, ...
        'ftol',0,'gtol',tol,'cg_preconditioner','diag','lbfgs_mem',20);
    [~,out_tmap_cg] = lr_2mproj(x0,A,b,mu,tmapOpts,2);
    [~,out_tmap_lbfgs] = lr_2mproj(x0,A,b,mu,tmapOpts,1);

    lambda = ones(n,1);
    funObj = @(w)LogisticLoss(w,A,b);
    pssgbOpts = struct('verbose',0,'maxIter',1000,'optTol',tol, ...
        'lossType','logistic','objectiveScale',1/m,'quadraticInit',1, ...
        'stopCriterion','proxResidual','mvpPerEval',[1,1], ...
        'cgTol',1e-1,'cgMaxIter',10,'cgAdapt',1,'corrections',20);
    pssgbOpts.innerSolver = 'cg';
    [~,out_pssgb_cg] = L1General2_PSSgb(funObj,x0,lambda,pssgbOpts);
    pssgbOpts.innerSolver = 'lbfgs';
    [~,out_pssgb_lbfgs] = L1General2_PSSgb(funObj,x0,lambda,pssgbOpts);

    altOpts = struct('verbose',0,'maxit',1000,'ls',1,'bb',1,'alpha0',1, ...
        'ftol',0,'gtol',tol);
    [~,out_altn] = lr_newtonacc(x0,A,b,mu,altOpts);

    data.A = A; data.b = b;
    model.loss = 'logistic'; model.penalty = 'ell1';
    model.regpara = mu; model.eps = tol;
    irpnOpts = struct('maxit',100,'maxitsub',100,'mu',1e-6,'eta',0.5, ...
        'rho',0.5,'beta',0.25,'sigma',0.25);
    [~,~,~,~,time_irpn,~,~,out_irpn] = alg_rpn(x0,data,model,irpnOpts);

    time = [out_tmap_cg.tt; out_tmap_lbfgs.tt; out_pssgb_cg.runtime; ...
        out_pssgb_lbfgs.runtime; out_altn.tt; time_irpn];
    N_A = [out_tmap_cg.mvp_total; out_tmap_lbfgs.mvp_total; ...
        out_pssgb_cg.mvp_total; out_pssgb_lbfgs.mvp_total; ...
        out_altn.mvp_total; out_irpn.mvpvec(end)];
    method = methodNames;
    T = table(method,time,N_A);
    disp(T);

    name = erase(datasets{id},'.binary');
    writetable(T,[name,'-results.csv']);
    fid = fopen([name,'-results.tex'],'w');
    fprintf(fid,'\\begin{tabular}{lrr}\n\\hline\nMethod & Time (s) & $N_A$ \\\\\n');
    for i=1:height(T)
        fprintf(fid,'%s & %.2f & %d \\\\\n',T.method{i},T.time(i),T.N_A(i));
    end
    fprintf(fid,'\\hline\n\\end{tabular}\n');
    fclose(fid);
end
