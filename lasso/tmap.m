function [x, out] = tmap(A,b,N,mu,opts)
%TMAP LASSO TMAP with displacement-based sufficient-decrease evaluation.
% One A*step per line-search trial; accepted residuals are reused for gradients.
% opts.residualRefreshPeriod=0 (default) disables extra A*x refresh products.
% opts.inner_solver is 'cg' (default) or 'lbfgs'.

tt = tic;
%%-------------------------------------------------------------------------
if nargin < 5; opts = []; end
if ~isfield(opts,'x0');    opts.x0  = zeros(N,1);   end
if ~isfield(opts,'tol');   opts.tol = 1e-6;     end
if ~isfield(opts,'maxit'); opts.maxit = 100; end
if ~isfield(opts,'inner_solver'); opts.inner_solver = 'cg'; end
if ~isfield(opts,'lbfgs_mem'); opts.lbfgs_mem = 20; end
if ~isfield(opts,'lbfgs_curvature_tol'); opts.lbfgs_curvature_tol = 1e-8; end
if ~isfield(opts,'cont');     opts.cont = 1;   end
if ~isfield(opts,'cont_max'); opts.cont_max = 10;   end
if ~isfield(opts,'crit'); opts.crit = 1;   end
if ~isfield(opts,'stopCriterion'); opts.stopCriterion = 'proxResidual'; end
if ~isfield(opts,'fReference'); opts.fReference = []; end
if ~isfield(opts,'recordFcn'); opts.recordFcn = []; end
if ~isfield(opts,'allowForcedAcceptance'); opts.allowForcedAcceptance = false; end
if ~isfield(opts,'residualRefreshPeriod'); opts.residualRefreshPeriod = 0; end
validateattributes(opts.residualRefreshPeriod,{'numeric'}, ...
    {'scalar','real','finite','integer','nonnegative'});

if isstring(opts.inner_solver); opts.inner_solver = char(opts.inner_solver); end
if ~ischar(opts.inner_solver) || ...
        ~any(strcmpi(opts.inner_solver,{'cg','lbfgs'}))
    error('opts.inner_solver must be ''cg'' or ''lbfgs''.');
end
opts.stopCriterion = resolveStopCriterion(opts.stopCriterion);
if strcmp(opts.stopCriterion,'relativeFunctionGap') && ...
        (isempty(opts.fReference) || ~isscalar(opts.fReference) || ~isfinite(opts.fReference))
    error('opts.fReference must be a finite scalar for relativeFunctionGap stopping.');
end
useLBFGS = strcmpi(opts.inner_solver,'lbfgs');
if ~isscalar(opts.lbfgs_mem) || opts.lbfgs_mem < 1 || ...
        opts.lbfgs_mem ~= floor(opts.lbfgs_mem)
    error('opts.lbfgs_mem must be a positive integer.');
end
if ~isscalar(opts.lbfgs_curvature_tol) || ...
        ~isfinite(opts.lbfgs_curvature_tol) || opts.lbfgs_curvature_tol <= 0
    error('opts.lbfgs_curvature_tol must be a finite positive scalar.');
end
if useLBFGS
    % Cache MEX availability for all subsequent L-BFGS directions.
    useCompiledLBFGS = exist('lbfgsC','file') == 3;
    if ~isfield(opts,'lbfgs_descent_fraction')
        opts.lbfgs_descent_fraction = 0.9;
    end
    if ~isscalar(opts.lbfgs_descent_fraction) || ...
            opts.lbfgs_descent_fraction < 0 || ...
            opts.lbfgs_descent_fraction > 1
        error('opts.lbfgs_descent_fraction must be in [0,1].');
    end
else
    if ~isfield(opts,'lambda'); opts.lambda = 1e-1; end
    if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 5; end
    if ~isfield(opts,'CG_tol'); opts.CG_tol = 1e-1; end
    if ~isfield(opts,'CG_adapt'); opts.CG_adapt = 1; end
end

x      = opts.x0;
tol    = opts.tol;

if opts.crit == 2
    fopt = opts.fopt;
end

maxit = opts.maxit;
epsilon = 1e-3;
sigma = 1e-1;
continuationProgressFloor = 1e-4;
out = struct();

if ~useLBFGS
    CG_tol = opts.CG_tol;
    CG_maxit = opts.CG_maxit;
    lambda = opts.lambda;
    regularizationPower = 0.7;
    if opts.CG_adapt; res_tol = 1e-3; end
end

% first computations
num_Ax      = 0;
num_ATy     = 0;
% No separate A'*b product is needed: gradients use the cached data residual.
residualRefreshCount = 0;
% Column nonzero counts: a product with A(:,I) costs sum(colNnz(I))/totalNnz
% of a full product with A.  Used for the dimension-weighted MVP count.
colNnz      = full(sum(A ~= 0,1));
totalNnz    = max(sum(colNnz),1);
weightedCG  = 0; % CG products in units of full products with A

x = sparse(x);
dataResidual = A*x-b;
num_Ax = num_Ax + 1;
funcx = 0.5*(dataResidual'*dataResidual) + mu*norm(x,1);

% compute residual function p = F_tau(x)
[Ftx, res, ~, grad] = comp_res(x);
num_ATy = num_ATy + 1;
resp = res;
epsilonk = min(epsilon,res);

% continuation
muf     = mu;

if opts.cont
    muredf = 0.2;
    mu = max(muf,muredf*norm(grad,'inf'));
    
    cont_max    = opts.cont_max;
    cont_count  = 1;
end

if opts.cont
    % Keep the state consistent after changing the continuation penalty.
    [Ftx,res] = proximalResidual(x,grad);
    epsilonk = min(epsilon,res);
    funcx = 0.5*(dataResidual'*dataResidual) + mu*norm(x,1);
    resp = res;
    resp_mu = res;
end

nr_CG = 0;
S = cell(0,1);
Y = cell(0,1);
Hdiag = 1;

out.fvec = zeros(maxit + 1,1);
out.nnzvec = zeros(maxit + 1,1);
out.num_Ax = zeros(maxit + 1,1);
out.num_ATy = zeros(maxit + 1,1);
out.mvp = zeros(maxit + 1,1);
out.timevec = zeros(maxit + 1,1);
if useLBFGS
    out.cg = NaN(maxit,1);
else
    out.cg = zeros(maxit,1);
end
out.lbfgs = zeros(maxit,1);
out.backtracks = zeros(maxit,1);
out.forcedAcceptance = false(maxit,1);
out.lineSearchTrials = zeros(maxit,1);
out.lineSearchDelta = zeros(maxit,1);
out.lineSearchRHS = zeros(maxit,1);
out.stepSize = zeros(maxit,1);
out.msg = 'iteration limit';
out.wsize = zeros(maxit,1);      % |I_minus| per iteration
out.wfrac = zeros(maxit,1);      % nnz(A(:,I_minus))/nnz(A) per iteration
out.mvp_weighted_vec = zeros(maxit + 1,1);
out.inner_solver = lower(opts.inner_solver);
if useLBFGS && useCompiledLBFGS
    out.lbfgsImplementation = 'lbfgsC';
elseif useLBFGS
    out.lbfgsImplementation = 'local';
else
    out.lbfgsImplementation = 'not applicable';
end
out.fvec(1) = targetObjective(x,dataResidual);
out.nnzvec(1) = nnz(x);
out.num_Ax(1) = num_Ax;
out.num_ATy(1) = num_ATy;
out.mvp(1) = num_Ax + num_ATy;
out.mvp_weighted_vec(1) = out.mvp(1);
out.timevec(1) = toc(tt);
if ~isempty(opts.recordFcn)
    opts.recordFcn(x,out.fvec(1),out.timevec(1),out.mvp(1),0);
end

completedIterations = 0;
for iter = 1:maxit
    if stoppingSatisfied(x,funcx,res)
        out.msg = 'optimal';
        break;
    end
    if opts.residualRefreshPeriod > 0 && mod(iter,opts.residualRefreshPeriod)==0
        dataResidual = A*x-b;
        num_Ax = num_Ax + 1;
        residualRefreshCount = residualRefreshCount + 1;
        [Ftx,res,~,grad] = comp_res(x);
        num_ATy = num_ATy + 1;
        epsilonk = min(epsilon,res);
        funcx = 0.5*(dataResidual'*dataResidual) + mu*norm(x,1);
        resp = res;
        if opts.cont; resp_mu = res; end
        if stoppingSatisfied(x,funcx,res)
            out.msg = 'optimal';
            current = completedIterations+1;
            out.fvec(current) = targetObjective(x,dataResidual);
            out.num_Ax(current) = num_Ax;
            out.num_ATy(current) = num_ATy;
            out.mvp(current) = num_Ax+num_ATy;
            out.mvp_weighted_vec(current) = ...
                (num_Ax+num_ATy-2*nr_CG)+weightedCG;
            out.timevec(current) = toc(tt);
            break;
        end
    end
    xp = x;
    gradp = grad;
    
    index_max = (x > epsilonk) | (0 <= x & x <= epsilonk & grad <= -mu);
    index_min = (x < -epsilonk) | (x >= -epsilonk & x <= 0 & grad >= mu);
    I_minus = logical(index_max + index_min);

    w = zeros(N,1);
    w(index_max) = mu;
    w(index_min) = -mu;

    gw = grad + w;

    % Build the working-set direction with either truncated CG or L-BFGS.
    d   = grad;
    out.wsize(iter) = nnz(I_minus);
    out.wfrac(iter) = sum(colNnz(I_minus))/totalNnz;
    if useLBFGS
        if any(I_minus)
            [d(I_minus),nPairs] = lbfgsDirection( ...
                gw(I_minus),S,Y,I_minus,Hdiag);
            linesearch_I_minus = opts.lbfgs_descent_fraction* ...
                max(gw(I_minus)'*d(I_minus),0);
        else
            nPairs = 0;
            linesearch_I_minus = 0;
        end
        out.lbfgs(iter) = nPairs;
    else
        gamma_vec = Ftx;
        gamma_vec(I_minus) = gw(I_minus);
        sig = lambda*(min(norm(gamma_vec),1)^regularizationPower);
        Aminus = A(:,I_minus);
        [d(I_minus),iterd,~,~,~,~,~] = ...
            mypcgw_tmap(Aminus,gw(I_minus),CG_tol,CG_maxit,[],0,sig);
        nr_CG = nr_CG + iterd;
        num_Ax = num_Ax + iterd;
        num_ATy = num_ATy + iterd;
        weightedCG = weightedCG + 2*iterd*out.wfrac(iter);
        out.cg(iter) = iterd;
        linesearch_I_minus = (1-CG_tol)*sig*norm(d(I_minus))^2;
    end
    %----------------------------------------------------------------------
    
    % Evaluate the same sufficient-decrease condition via an objective
    % increment. Keep the base residual fixed across rejected trials.
    t = 1;
    [xz, nrm_dx_plus] = proj(x, d, t);
    xz = sparse(xz);
    nls = 0;
    while true
        if nnz(xz-x)==0
            error('TMAP:LineSearchStall', ...
                'No representable trial satisfies sufficient decrease.');
        end
        [objectiveChange,trialResidual] = ...
            lasso_trial_increment(A,x,xz,dataResidual,mu);
        num_Ax = num_Ax + 1;
        decreaseBound = -sigma*t*linesearch_I_minus - sigma/t*nrm_dx_plus^2;
        sufficientDecrease = isfinite(objectiveChange) && ...
            isfinite(decreaseBound) && objectiveChange <= decreaseBound;
        forced = opts.allowForcedAcceptance && ~useLBFGS && nls==5;
        if sufficientDecrease || (forced && isfinite(objectiveChange))
            break;
        end
        if nls >= 500 || t < realmin
            error('TMAP:LineSearchStall', ...
                'No representable trial satisfies sufficient decrease.');
        end
        t = 0.2*t;
        nls = nls + 1;
        [xz, nrm_dx_plus] = proj(x, d, t);
        xz = sparse(xz);
    end
    out.backtracks(iter) = nls;
    out.lineSearchTrials(iter) = nls+1;
    out.lineSearchDelta(iter) = objectiveChange;
    out.lineSearchRHS(iter) = decreaseBound;
    out.stepSize(iter) = t;
    out.forcedAcceptance(iter) = ~sufficientDecrease;
    dataResidual = trialResidual;
    funcxz = 0.5*(dataResidual'*dataResidual) + mu*norm(xz,1);
    [Ftx, resxz, ~, gradxz] = comp_res(xz);
    num_ATy = num_ATy + 1;
    x = xz; res = resxz; grad = gradxz; epsilonk = min(epsilon, res);
    completedIterations = iter;
    if useLBFGS
        [S,Y,Hdiag] = updateLBFGSHistory( ...
            S,Y,Hdiag,x-xp,grad-gradp,opts.lbfgs_mem, ...
            opts.lbfgs_curvature_tol);
    end
    funcx = funcxz;
    out.fvec(iter + 1) = targetObjective(x,dataResidual);
    out.nnzvec(iter + 1) = nnz(x);
    out.num_Ax(iter + 1) = num_Ax;
    out.num_ATy(iter + 1) = num_ATy;
    out.mvp(iter + 1) = num_Ax + num_ATy;
    out.mvp_weighted_vec(iter + 1) = (num_Ax + num_ATy - 2*nr_CG) + weightedCG;
    out.timevec(iter + 1) = toc(tt);
    if ~isempty(opts.recordFcn)
        opts.recordFcn(x,out.fvec(iter+1),out.timevec(iter+1),out.mvp(iter+1),iter);
    end
             
    % stopping criteria
    if stoppingSatisfied(x,funcx,res)
        out.msg = 'optimal';
        break;
    end
       
        % evaluate the quality of the last iterate
        if opts.cont
            goodIter = (cont_count >= cont_max) ||  ...
             ((res <= 0.5*resp) || ...
             (res <= max(0.1*resp_mu, continuationProgressFloor)));
        end
            
        % continuation: update mu
        previousMu = mu;
        if opts.cont
            if goodIter
                muredf = 0.2;
                mu          = max(muf,muredf*norm(grad,'inf'));
                                
                if mu > muf
                    cont_count      = 1;
                else
                    cont_count      = cont_count + 1;
                end
            else
                cont_count = cont_count + 1;
            end
        end
        
        if mu ~= previousMu
            [Ftx,res] = proximalResidual(x,grad);
            epsilonk = min(epsilon,res);
            funcx = 0.5*(dataResidual'*dataResidual) + mu*norm(x,1);
        end
        if opts.cont; resp_mu = res; end

     %   update CG parameters if final mu is reached
        if ~useLBFGS && opts.CG_adapt && (mu <= (1+1e-10)*muf)
            if res < res_tol
                CG_tol     = max(0.1*CG_tol,1e-7);
                res_tol    = 0.1*res_tol;
            end
            
            if res < 1e-3
                CG_maxit   = min(max(CG_maxit + 7,3),N);
            elseif res < 1e-2
                CG_maxit   = 50;
            elseif res < 1e-1
                CG_maxit   = 30;
            else
                CG_maxit   = 25;
            end
        end

        resp = res;
    
end %end outer loop
% Continuation may reach the target penalty on the final allowed iteration.
% Report that state consistently even when there is no next loop entry.
if stoppingSatisfied(x,funcx,res)
    out.msg = 'optimal';
end
iter = completedIterations;
out.time = toc(tt);
out.runtime = out.time;

out.backtracks = out.backtracks(1:iter);
out.lineSearchTrials = out.lineSearchTrials(1:iter);
out.lineSearchDelta = out.lineSearchDelta(1:iter);
out.lineSearchRHS = out.lineSearchRHS(1:iter);
out.stepSize = out.stepSize(1:iter);
out.residualRefreshCount = residualRefreshCount;
out.dataResidual = dataResidual;
out.lineSearchEvaluation = 'objective increment';
out.forcedAcceptance = out.forcedAcceptance(1:iter);
out.iter = iter;
out.res  = res;
out.f = funcx;

out.nr_CG = nr_CG;
out.cg = out.cg(1:iter);
out.lbfgs = out.lbfgs(1:iter);
out.fvec = out.fvec(1:iter + 1);
out.nnzvec = out.nnzvec(1:iter + 1);
out.num_Ax = out.num_Ax(1:iter + 1);
out.num_ATy = out.num_ATy(1:iter + 1);
out.mvp = out.mvp(1:iter + 1);
out.wsize = out.wsize(1:iter);
out.wfrac = out.wfrac(1:iter);
out.mvp_weighted_vec = out.mvp_weighted_vec(1:iter + 1);
out.timevec = out.timevec(1:iter + 1);
out.num_Ax_total = num_Ax;
out.num_ATy_total = num_ATy;
out.mvp_total = num_Ax + num_ATy;
% Full products counted once; CG products weighted by the working-set share.
out.mvp_weighted = (num_Ax + num_ATy - 2*nr_CG) + weightedCG;
out.Acalls = out.mvp_total;

    function ss = shringkage(xx,mumu)
        ss = sign(xx).*max(abs(xx)-mumu,0);
    end

    function [Ftx, res, yx, grad] = comp_res(xx)
        grad = A'*dataResidual;
        [Ftx,res,yx] = proximalResidual(xx,grad);
    end

    function [Ftx,res,yx] = proximalResidual(xx,gg)
        yx = xx-gg;
        Ftx = xx-shringkage(yx,mu);
        res = norm(Ftx);
    end

    function fv = targetObjective(xx,rr)
        fv = muf*norm(xx,1) + 0.5*(rr'*rr);
    end

    function satisfied = stoppingSatisfied(xx,objective,proxResidual)
        atTargetPenalty = mu <= (1+1e-10)*muf;
        if strcmp(opts.stopCriterion,'relativeFunctionGap')
            satisfied = atTargetPenalty && ...
                (objective-opts.fReference)/max(1,abs(opts.fReference)) <= tol;
        elseif strcmp(opts.stopCriterion,'relativeKKT')
            % Relative KKT residual of Li, Sun and Toh (SSNAL).
            satisfied = atTargetPenalty && ...
                proxResidual/(1+norm(xx)+norm(dataResidual)) <= tol;
        elseif opts.crit == 2
            satisfied = atTargetPenalty && ...
                (objective-fopt)/max(1,abs(fopt)) <= tol;
        else
            satisfied = atTargetPenalty && proxResidual <= tol;
        end
    end

    function [y, nrm_dx_plus] = proj(xx, d, t)
        z = xx - t*d;
        
        y = shringkage(z,t*mu);

        y(I_minus) = 0;
        xx(I_minus) = 0;
        nrm_dx_plus = norm(y - xx);

        y(index_max) = max(z(index_max),0);
        y(index_min) = min(z(index_min),0);

        
    end

    function [S,Y,Hdiag] = updateLBFGSHistory( ...
            S,Y,Hdiag,s,y,maxPairs,curvatureTol)
        ys = y'*s;
        yy = y'*y;
        if any(~isfinite(s)) || any(~isfinite(y)) || ...
                ~isfinite(ys) || ~isfinite(yy)
            return;
        end
        % Use a relative curvature test so small, well-aligned steps near
        % convergence are not rejected merely because |s' * y| is tiny.
        curvatureScale = max(norm(s)*norm(y),realmin);
        curvatureOK = ys > curvatureTol*curvatureScale && yy > 0;
        if curvatureOK
            Hdiag = ys/yy;
        end
        if numel(S) >= maxPairs
            S = S(2:end);
            Y = Y(2:end);
        end
        % Dense history avoids repeated sparse logical indexing on working sets.
        S{end+1,1} = full(s);
        Y{end+1,1} = y;
    end

    function [d,nPairs] = lbfgsDirection(g,S,Y,mask,Hdiag)
        nPairs = 0;
        valid = zeros(1,numel(S));
        for j = 1:numel(S)
            sj = S{j}(mask);
            yj = Y{j}(mask);
            curvatureScale = max(norm(sj)*norm(yj),realmin);
            if isfinite(yj'*sj) && ...
                    yj'*sj > opts.lbfgs_curvature_tol*curvatureScale
                nPairs = nPairs + 1;
                valid(nPairs) = j;
            end
        end
        valid = valid(1:nPairs);
        if nPairs == 0
            d = Hdiag*g;
            return;
        end

        Swork = zeros(numel(g),nPairs);
        Ywork = zeros(numel(g),nPairs);
        for j = 1:nPairs
            Swork(:,j) = S{valid(j)}(mask);
            Ywork(:,j) = Y{valid(j)}(mask);
        end
        if useCompiledLBFGS
            d = lbfgsC(g,Swork,Ywork,Hdiag);
        else
            d = localLBFGS(g,Swork,Ywork,Hdiag);
        end
    end

    function d = localLBFGS(g,S,Y,Hdiag)
        nPairs = size(S,2);
        q = g;
        alpha = zeros(nPairs,1);
        rho = zeros(nPairs,1);
        for j = nPairs:-1:1
            rho(j) = 1/(Y(:,j)'*S(:,j));
            alpha(j) = rho(j)*(S(:,j)'*q);
            q = q-alpha(j)*Y(:,j);
        end
        d = Hdiag*q;
        for j = 1:nPairs
            beta = rho(j)*(Y(:,j)'*d);
            d = d+S(:,j)*(alpha(j)-beta);
        end
    end

    function criterion = resolveStopCriterion(criterion)
        if isstring(criterion); criterion = char(criterion); end
        if ~ischar(criterion)
            error('opts.stopCriterion must be ''proxResidual'' or ''relativeFunctionGap''.');
        end
        key = lower(strtrim(criterion));
        if any(strcmp(key,{'proxresidual','prox-residual','prox_residual'}))
            criterion = 'proxResidual';
        elseif any(strcmp(key,{'relativefunctiongap','relative-function-gap', ...
                'relative_function_gap','functiongap','function-gap'}))
            criterion = 'relativeFunctionGap';
        elseif any(strcmp(key,{'relativekkt','relative-kkt','relative_kkt','kkt'}))
            criterion = 'relativeKKT';
        else
            error('opts.stopCriterion must be ''proxResidual'', ''relativeKKT'' or ''relativeFunctionGap''.');
        end
    end


end
