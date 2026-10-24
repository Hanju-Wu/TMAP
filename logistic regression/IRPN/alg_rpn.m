function [funv_array, xopt, resi_array, num_iter, cput, ssa, dimension, stats] = alg_rpn( x0, data, model, opts)

% This function uses IRPN to solve l1-regularized logistic regression
% opts.maxTime (seconds, default Inf) limits wall time; stats.timedOut records timeout.
% problem. The decription of the algorithm can be found in our paper:

% M.-C. Yue, Z. Zhou, A. M.-C. So:

% "A Family of Inexact SQA Methods for Non-Smooth Convex Minimization with 
% Provable Convergence Guarantees Based on the Luo-Tseng Error Bound
% Property"

% Inputs:
%   1. x0: initial point
%   2. data: samples and labels
%   3. model: regpara, penalty, loss, eps
%   4. opts: IRPN specific parameters:
%           (a). opts.maxit: maximum number of outer iterations
%           (b). opts.maxitsub: maximum number of inner iterations
%           (c). opts.mu = 1e-6
%           (d). opts.eta = 0.5;
%           (e). opts.rho = 0, 0.5, or 1.


% Outputs:
%   1. funv_array: the array of function values of each outer iterate
%   2. resi_array: the array of residual values of each outer iterate
%   3. num_iter: the array of every number of inner iterations
%   4. cput: cpu time
%   5. ssa: the array of step sizes of each iteration

if ~isfield(opts, 'opt_dim'); opts.opt_dim = 0; end
if ~isfield(opts, 'maxTime'); opts.maxTime = Inf; end
validateattributes(opts.maxTime,{'numeric'},{'real','scalar','nonnegative','nonnan'});


x = x0;

A = data.A;
b = data.b;
m = size(A,1);
eps = model.eps;
useGap = isfield(model,'f_ref') && ~isempty(model.f_ref) && ...
    isfield(model,'gap_tol') && ~isempty(model.gap_tol);
rp = model.regpara;

sigma = opts.sigma;
beta = opts.beta;
maxit = opts.maxit;
maxitsub = opts.maxitsub;
eta = opts.eta;
rho = opts.rho;

% record stepsizes, function values, residuals and number of each inner
% iterations
ssa = zeros(1,maxit);
funv_array = [];
resi_array = zeros(1, maxit);
num_iter = zeros(1, maxit);
dimension = sum(abs(x)>0);
mu = opts.mu;

fprintf('IRPN rho = %1.1f is working...\n',rho);

tstart = tic;
timedOut = false;
shouldStop = [];
if isfinite(opts.maxTime)
    shouldStop = @() toc(tstart) >= opts.maxTime;
end
mvp = 0;
time_array = zeros(1,maxit);
mvp_array = zeros(1,maxit);
nupdates = 0;
nrecorded = 0;

for k=1:maxit-1
    % mu = opts.mu;
    %
    % Computing function value, gradient, hessian diagonal, and residual at x
    %
    
    funv_s = loss_smv(x,data);
    mvp = mvp + 1;
    funv_n = nsm_funv(x,model);
    funv = funv_s + funv_n;
    funv_array = [funv_array, funv];
    
    grad = loss_smg(x,data);
    mvp = mvp + 2;
    Dx = loss_smh_diag(x,data);
    mvp = mvp + 1;
    
    resi = norm(x - prox_oper_vec(x-grad,model,1),2);
    resi_array(k) = resi;
    % Record cost after evaluating this same iterate's residual.
    time_array(k) = toc(tstart);
    mvp_array(k) = mvp;
    nrecorded = k;
    
%     fprintf('At iteration %d of RPN_CD, residual %f\n', k, resi);
    if (useGap && relative_objective_gap(funv, model.f_ref)<=model.gap_tol) || ...
            (~useGap && resi<eps)
        fprintf('RPN_CD achieves required resi accuracy\n');
        break;
    end
    if ~isempty(shouldStop) && shouldStop()
        timedOut = true;
        break;
    end
    


    
    %
    % perform regularized Newton with line search
    %
    
    mu = mu*max(resi, (resi)^rho);    % Hessian regularization
    eps_in = eta*min(resi, resi^(1+rho));    % stopping cri for inner loop
    
    s = x;
    As = A*x;
    q =  A'*(Dx.*(As)) + mu*x - grad;
    mvp = mvp + 2;
    
    % The subproblem then can be written as
    %       min_x 0.5*s^T(A'*Dx*A+mu*I)s - q^Ts + lambda*||s||_1.
    resi_in = resi;
    
    for i=1:maxitsub
        if ~isempty(shouldStop) && shouldStop()
            timedOut = true;
            break;
        end
        if resi_in<eps_in
%             fprintf('inner finished with resi_in = %f\n', resi_in);
            break;
        end
        
        [s,As] = ripn_cordmin(s, A, Dx, As, q, rp, mu); % perform a cycle of coordinate minimization

        resi_in = norm(s - prox_oper_vec(s - (A'*(Dx.*(As)) + mu*s - q), model, 1), 2);
        mvp = mvp + 1;
        
        if mod(i-1,5)==0
            eAx = exp(As);
            grad_t = A'*(eAx./(1+eAx) - b)/m; 
            mvp = mvp + 1;
            resi_t = norm(s - prox_oper_vec(s-grad_t,model,1),2);
            if (~useGap && resi_t<eps)
                 fprintf('inner finished with resi_in = %f\n', resi_in);
                break;
            end
        end
    end
    if timedOut || (~isempty(shouldStop) && shouldStop())
        timedOut = true;
        break;
    end
    
%========================================================================================
%   Below is pure Matlab implementation. It is much slower than Mex file
%   implementation. 
% 
%     p = size(A,2);
%     s = x;      % s is initial point of inner loop;
%     Ax = A*x;
%     As = Ax;
%     adax = A'*(Dx.*Ax);
%     hess_diag = zeros(p,1);
%     for i=1:p
%         hess_diag(i) = dot(A(:,i).^2, Dx);
%     end
%     
%     resi_in = resi;
%     fprintf('resi_in = %f\n', resi);
%     
%     for i=1:maxitsub
%         
%         if resi_in<eps_in
%             %fprintf('inner finished with resi_in = %f\n', resi_in);
%             break;
%         end
%         
%         for ind=1:p
%             aii = hess_diag(ind) + mu;
%             alpha = dot(A(:,ind), Dx.*As) - hess_diag(ind)*s(ind) + grad(ind) - adax(ind) - mu*x(ind);
%             s_p = soft_skg(-alpha, rp)/aii;
%             As = As + A(:,ind)*(s_p - s(ind));
%             s(ind) = s_p;
%         end
%         
%         adas = A'*(Dx.*As);
%         resi_in = norm(s - soft_skg(s - (grad + adas - adax + mu*(s - x)), rp), 2)
%     end
%========================================================================================
    
    
    % search direction d = s-x, and perform backtracking line search
    d = s - x;
    ss = 1;
    disc = ss*dot(grad, d) + nsm_funv(s,model) - funv_n;
    funv_new = loss_smv(s,data) + nsm_funv(s,model);
    mvp = mvp + 1;
    while(funv_new > funv_s + funv_n + sigma*disc)
        if ~isempty(shouldStop) && shouldStop()
            timedOut = true;
            break;
        end
        ss = ss * beta;
        s = x + ss*d;
        disc = ss*dot(grad, d) + nsm_funv(s,model) - funv_n;
        funv_new = loss_smv(s,data) + nsm_funv(s,model);
        mvp = mvp + 1;
    end
    if timedOut || (~isempty(shouldStop) && shouldStop())
        % The subproblem/line-search candidate has not been accepted.
        timedOut = true;
        break;
    end
    
    % if abs(funv-funv_new)<1e-8 
    %     fprintf('RPN_CD achieves required funv accuracy\n');
    %     break;
    % end

    ssa(k) = ss;
    num_iter(k) = i;
    
    x = s;
    nupdates = nupdates + 1;
    dim = sum(abs(x)>0);
    dimension = [dimension, dim];
%     fprintf('Iteration %d is finished with %d number of inner iterations\n\n',k,i-1);
end

xopt = x;

% If the iteration limit follows an update, its final point has not yet
% been evaluated. Include that point and the cost of checking its residual.
% On convergence, the loop has already recorded it; do not evaluate twice.
if nrecorded == nupdates
    nrecorded = nrecorded + 1;
    funv_array(nrecorded) = loss_smv(x,data) + nsm_funv(x,model);
    grad = loss_smg(x,data);
    mvp = mvp + 3;
    resi_array(nrecorded) = norm(x - prox_oper_vec(x-grad,model,1),2);
    time_array(nrecorded) = toc(tstart);
    mvp_array(nrecorded) = mvp;
end
cput = toc(tstart);
funv_array = reshape(funv_array(1:nrecorded),[],1);
resi_array = resi_array(1:nrecorded);
time_array = time_array(1:nrecorded);
mvp_array = mvp_array(1:nrecorded);
if isfield(model,'f_ref'); f_ref = model.f_ref; else; f_ref = []; end
stats = struct('fvec',funv_array,'timevec',time_array(:), ...
    'mvpvec',mvp_array(:),'res',resi_array(:), ...
    'gapvec',relative_objective_gap(funv_array,f_ref), ...
    'iter',nupdates,'runtime',cput,'mvp_total',mvp, ...
    'maxTime',opts.maxTime,'timedOut',timedOut);

fprintf('IRPN rho = %1.1f is finished.\n\n',rho);

end
    
