function [x, out] = tmap(A,b,N,mu,opts)
%TMAP Set opts.inner_solver to 'cg' (default) or 'lbfgs'.

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
    if ~isfield(opts,'lbfgs_descent_fraction')
        opts.lbfgs_descent_fraction = 0.9;
    end
    if ~isscalar(opts.lbfgs_descent_fraction) || ...
            opts.lbfgs_descent_fraction < 0 || ...
            opts.lbfgs_descent_fraction > 1
        error('opts.lbfgs_descent_fraction must be in [0,1].');
    end
else
    if ~isfield(opts,'lambda'); opts.lambda = 1e-4; end
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
Atb         = A'*(b);
num_ATy     = num_ATy + 1;

x = sparse(x);
[funcx, Ax] = func(x);
num_Ax = num_Ax + 1;

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

if opts.cont; resp_mu = res; end

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
out.inner_solver = lower(opts.inner_solver);
if useLBFGS && exist('lbfgsC','file') == 3
    out.lbfgsImplementation = 'lbfgsC';
elseif useLBFGS
    out.lbfgsImplementation = 'local';
else
    out.lbfgsImplementation = 'not applicable';
end
out.fvec(1) = targetObjective(x,Ax);
out.nnzvec(1) = nnz(x);
out.num_Ax(1) = num_Ax;
out.num_ATy(1) = num_ATy;
out.mvp(1) = num_Ax + num_ATy;
out.timevec(1) = toc(tt);

for iter = 1:maxit
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
        out.cg(iter) = iterd;
        linesearch_I_minus = (1-CG_tol)*sig*norm(d(I_minus))^2;
    end
    %----------------------------------------------------------------------
    
    % line search
    t = 1;
    [xz, nrm_dx_plus] = proj(x, d, t);
    xz = sparse(xz);

    nls = 0;
    while 1
        [funcxz, Ax] = func(xz);
        num_Ax = num_Ax + 1;
        if  funcxz - funcx <= -sigma*t*linesearch_I_minus - sigma/t*nrm_dx_plus^2 %|| nls == 5
            break;
        end
            
        t = 0.2*t; nls = nls + 1;
        [xz, nrm_dx_plus] = proj(x, d, t);
        xz = sparse(xz);
    end

    [Ftx, resxz, ~, gradxz] = comp_res(xz);
    num_ATy = num_ATy + 1;
    x = xz; res = resxz; grad = gradxz; epsilonk = min(epsilon, res);
    if useLBFGS
        [S,Y,Hdiag] = updateLBFGSHistory( ...
            S,Y,Hdiag,x-xp,grad-gradp,opts.lbfgs_mem, ...
            opts.lbfgs_curvature_tol);
    end
    funcx = funcxz;
    out.fvec(iter + 1) = targetObjective(x,Ax);
    out.nnzvec(iter + 1) = nnz(x);
    out.num_Ax(iter + 1) = num_Ax;
    out.num_ATy(iter + 1) = num_ATy;
    out.mvp(iter + 1) = num_Ax + num_ATy;
    out.timevec(iter + 1) = toc(tt);
             
    % stopping criteria
    if strcmp(opts.stopCriterion,'relativeFunctionGap')
        cstop = (mu <= (1+1e-10)*muf) && ...
            ((funcx - opts.fReference)/max(1,abs(opts.fReference)) <= tol);
    elseif opts.crit == 2
        cstop = (mu <= (1+1e-10)*muf) && ...
            ((funcx - fopt)/max(1,abs(fopt)) <= tol);
    else
        cstop = (mu <= (1+1e-10)*muf) && (res <= tol);
    end
    
    if cstop
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
out.time = toc(tt);
out.runtime = out.time;

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
out.timevec = out.timevec(1:iter + 1);
out.num_Ax_total = num_Ax;
out.num_ATy_total = num_ATy;
out.mvp_total = num_Ax + num_ATy;
out.Acalls = out.mvp_total;

    function ss = shringkage(xx,mumu)
        ss = sign(xx).*max(abs(xx)-mumu,0);
    end

    function [Ftx, res, yx, grad] = comp_res(xx)
        grad   = A'*(Ax) - Atb;
        yx     = xx - grad;
        Ftx    = xx - shringkage(yx,mu);
        res    = norm(Ftx); 
    end

    function [fv,Ax] = func(xx)
        Ax = A*xx;
       fv = mu*norm(xx,1) + 0.5*norm(Ax-b)^2; 
    end

    function fv = targetObjective(xx,Axx)
        fv = muf*norm(xx,1) + 0.5*norm(Axx-b)^2;
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
        S{end+1,1} = s;
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
        if exist('lbfgsC','file') == 3
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
        else
            error('opts.stopCriterion must be ''proxResidual'' or ''relativeFunctionGap''.');
        end
    end


end
