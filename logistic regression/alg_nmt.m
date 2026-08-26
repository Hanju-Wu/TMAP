function [funv_array, xopt, resi_array, num_iter, cput, ssa, stats] = alg_nmt( x0, data, model, opts )

% This function uses SpaRSA to solve l1-regularized logistic regression
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"


% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: SpaRSA specific parameters:
%           (a). opts.maxit: maximum number of iterations
%           (b). opts.c = 1e-4;
%           (c). opts.M = 5;
%           (d). opts.ssu = 1e+8;
%           (e). opts.ssl = 1e-8;


% Outputs:
%   1. funv_array: the array of function values of each iterate
%   2. resi_array: the array of residual values of each iterate
%   3. num_iter: the number of iterations
%   4. cput: cpu time
%   5. ssa: the array of step-sizes used in each iteration


eps = model.eps;
useGap = isfield(model,'f_ref') && ~isempty(model.f_ref) && ...
    isfield(model,'gap_tol') && ~isempty(model.gap_tol);
beta = opts.beta;
maxit = opts.maxit;
ssu = opts.ssu;
ssl = opts.ssl;
c = opts.c;
M = opts.M;

dis_freq = Inf;
fprintf('SpaRSA is working...\n');

% record stepsizes, function values, and residuals
ssa = zeros(1, maxit);
funv_array = zeros(1,maxit);
resi_array = zeros(1,maxit);

tstart = tic;
mvp = 0;
time_array = zeros(1,maxit+1);
mvp_array = zeros(1,maxit+1);

k = 0;
x = x0;
ss0 = max(ssl,min(ssu,1));
funv_array(1) = loss_smv(x,data) + nsm_funv(x,model);
grad = loss_smg(x,data);
mvp = mvp + 3;
resi = norm(x - prox_oper_vec(x-grad,model,1),2);
resi_array(1) = resi;
time_array(1) = toc(tstart);
mvp_array(1) = mvp;

F_cache = -inf(M,1);
F_cache(1) = funv_array(1);
Fmax = max(F_cache);

while k<maxit

    if mod(k,dis_freq)==0
        fprintf('This is teration %d of SpaRSA, and residual = %f\n', k, resi);
     end
        
    if (useGap && relative_objective_gap(funv_array(k+1), model.f_ref)<=model.gap_tol) || ...
            (~useGap && resi<eps)
        fprintf('SpaRSA achieves required accuracy\n');
        break;
    end
    
    %
    % perform a nonmonotone line search step
    %
    
    ss = ss0;
    while((ss>=ssl)&&(ss<=ssu))
        x_new = prox_oper_vec(x-ss*grad, model, ss);
        F_new = loss_smv(x_new,data) + nsm_funv(x_new,model);
        mvp = mvp + 1;
        if F_new <= Fmax - c/2*norm(x_new - x)^2
            break
        end
        ss = ss*beta;
    end
    
    ssa(k+1) = ss;    % record the step size of this iteration
    
    % BB step size
    dx = x_new - x;
    grad_new = loss_smg(x_new,data);
    mvp = mvp + 2;
    dg = grad_new - grad;
    chs1 = norm(dx)^2/abs(dx'*dg);
    chs2 = abs(dx'*dg)/norm(dg)^2;
    if mod(k,2) == 0
        ss0 = max(ssl,min(ssu,chs1));
    else
        ss0 = max(ssl,min(ssu,chs2));
    end
    
    % update iterate
    k = k+1;
    x = x_new;
    funv_array(k+1) = F_new;
    grad = grad_new;
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k+1) = resi;
    time_array(k+1) = toc(tstart);
    mvp_array(k+1) = mvp;
    F_cache(mod(k+1,M)+1,1) = F_new;
    Fmax = max(F_cache);
    
end

num_iter = k;
xopt = x;
cput = toc(tstart);
funv_array = funv_array(1:k+1);
resi_array = resi_array(1:k+1);
if isfield(model,'f_ref'); f_ref = model.f_ref; else; f_ref = []; end
stats = struct('fvec',funv_array,'timevec',time_array(1:k+1), ...
    'mvpvec',mvp_array(1:k+1), ...
    'gapvec',relative_objective_gap(funv_array,f_ref));

fprintf('SpaRSA is finished\n\n');

end



