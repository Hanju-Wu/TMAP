function [x, out] = lr_2mproj(x0, A, b, mu, opts, option)
%LR_2MPROJ Two-metric projection with selectable direction solver.
%   option = 1 uses L-BFGS on the working set; option = 2 uses the
%   original truncated-CG direction (default).  The same choice can be
%   supplied through opts.inner_solver = 'lbfgs' or 'cg'.
%   opts.cg_adaptive (default true) caps early CG solves at 10/15/20/25
%   iterations. Within opts.cg_unlimited_factor (default 100) times the
%   outer stopping tolerance, only the original working-set dimension cap
%   remains. out.cgMaxIter records the extra cap (Inf when removed).
%   opts.maxTime is an optional wall-time budget in seconds (default Inf).
%   Timeout returns the last accepted iterate and sets out.timedOut=true.
    if nargin < 5 || isempty(opts); opts = struct(); end
    if nargin < 6 || isempty(option)
        if isfield(opts, 'inner_solver')
            option = opts.inner_solver;
        elseif isfield(opts, 'option')
            option = opts.option;
        else
            option = 2;
        end
    end
    solver = normalizeSolver(option);
    useLBFGS = strcmp(solver, 'lbfgs');

    if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
    if ~isfield(opts, 'maxTime'); opts.maxTime = Inf; end
    validateattributes(opts.maxTime,{'numeric'},{'real','scalar','nonnegative','nonnan'});
    % Use a relative objective gap when a reference objective is supplied;
    % otherwise retain the proximal-gradient residual stopping criterion.
    if ~isfield(opts, 'f_ref'); opts.f_ref = []; end
    if ~isfield(opts, 'gap_tol'); opts.gap_tol = []; end
    % A fixed objective-decrease threshold can terminate high-accuracy
    % L-BFGS runs prematurely, so it is disabled unless explicitly set.
    if ~isfield(opts, 'ftol'); opts.ftol = 0; end
    if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
    if ~isfield(opts, 'verbose'); opts.verbose = 1; end
    if ~isfield(opts, 'alpha0'); opts.alpha0 = 1; end
    if ~isfield(opts, 'ls'); opts.ls = 1; end
    if ~isfield(opts, 'bb'); opts.bb = 0; end
    if ~isfield(opts, 'cg_preconditioner'); opts.cg_preconditioner = 'none'; end
    if ~isfield(opts, 'cg_adaptive'); opts.cg_adaptive = true; end
    if ~isfield(opts, 'cg_unlimited_factor'); opts.cg_unlimited_factor = 100; end
    if ~(islogical(opts.cg_adaptive) || isnumeric(opts.cg_adaptive)) || ...
            ~isscalar(opts.cg_adaptive) || ~ismember(opts.cg_adaptive,[0,1])
        error('opts.cg_adaptive must be true or false.');
    end
    if ~isnumeric(opts.cg_unlimited_factor) || ...
            ~isscalar(opts.cg_unlimited_factor) || ...
            ~isfinite(opts.cg_unlimited_factor) || opts.cg_unlimited_factor < 1
        error('opts.cg_unlimited_factor must be a finite scalar >= 1.');
    end
    % Match the L-BFGS configuration in PSSgb/TMP.
    if ~isfield(opts, 'lbfgs_mem'); opts.lbfgs_mem = 100; end
    if ~isscalar(opts.lbfgs_mem) || opts.lbfgs_mem < 1 || ...
            opts.lbfgs_mem ~= floor(opts.lbfgs_mem)
        error('opts.lbfgs_mem must be a positive integer.');
    end
    if ~isscalar(opts.ftol) || ~isfinite(opts.ftol) || opts.ftol < 0
        error('opts.ftol must be a nonnegative finite scalar.');
    end
    if isstring(opts.cg_preconditioner)
        opts.cg_preconditioner = char(opts.cg_preconditioner);
    end
    if ~ischar(opts.cg_preconditioner) || ...
            ~any(strcmpi(opts.cg_preconditioner,{'none','diag'}))
        error('opts.cg_preconditioner must be ''none'' or ''diag''.');
    end
    useDiagPreconditioner = strcmpi(opts.cg_preconditioner,'diag');
    
    out = struct();
    out.maxTime = opts.maxTime;
    out.timedOut = false;
    [m,n] = size(A);
    k = 0;
    
    x = x0;
    t = opts.alpha0;
    tt = tic;
    shouldStop = [];
    if isfinite(opts.maxTime)
        shouldStop = @() toc(tt) >= opts.maxTime;
    end
    num_Ax = 0;
    num_ATy = 0;
    S = cell(0,1);
    Y = cell(0,1);
    Hdiag = 1;
    
    Ax = A*x;
    num_Ax = num_Ax + 1;
    [f, g, p] = logisticState(A, b, Ax, x, mu, m);
    num_ATy = num_ATy + 1;
    
    res = norm(x - prox(x - g,mu),2);
    out.fvec = zeros(opts.maxit + 1, 1);
    out.dimension = zeros(opts.maxit + 1, 1);
    if useLBFGS
        out.cg = NaN(opts.maxit, 1);
    else
        out.cg = zeros(opts.maxit, 1);
    end
    out.cgMaxIter = NaN(opts.maxit, 1);
    out.lbfgs = zeros(opts.maxit, 1);
    out.nls = zeros(opts.maxit, 1);
    out.num_Ax = zeros(opts.maxit + 1, 1);
    out.num_ATy = zeros(opts.maxit + 1, 1);
    out.mvp = zeros(opts.maxit + 1, 1);
    out.timevec = zeros(opts.maxit + 1, 1);
    out.direction = solver;
    out.cgPreconditioner = opts.cg_preconditioner;
    % Resolve the L-BFGS kernel once per solve; CG skips the file lookup.
    useCompiledLBFGS = useLBFGS && exist('lbfgsC','file') == 3;
    if useCompiledLBFGS
        out.lbfgsImplementation = 'lbfgsC';
    elseif useLBFGS
        out.lbfgsImplementation = 'local';
    else
        out.lbfgsImplementation = 'not applicable';
    end
    out.res = zeros(opts.maxit + 1, 1);
    out.gap = zeros(opts.maxit + 1, 1);
    out.fvec(1) = f;
    out.dimension(1) = nnz(x);
    out.res(1) = res;
    if isempty(opts.f_ref)
        out.gap(1) = NaN;
    else
        out.gap(1) = relative_objective_gap(f, opts.f_ref);
    end
    out.num_Ax(1) = num_Ax;
    out.num_ATy(1) = num_ATy;
    out.mvp(1) = num_Ax + num_ATy;
    out.timevec(1) = toc(tt);
    
    
    sigma = 1e-1;
    epsilon = 1e-3;
    tao = 1e-1;
    delta = res;
    epsilonk = min(epsilon,delta);
    
    while k < opts.maxit && ((isempty(opts.f_ref) && res > opts.gtol) || ...
            (~isempty(opts.f_ref) && out.gap(k + 1) > opts.gap_tol))
        if ~isempty(shouldStop) && shouldStop()
            out.timedOut = true;
            break;
        end
        
        fp = f;
        xp = x;
    
        index_max = (x > epsilonk) | (0 <= x & x <= epsilonk & g <= -mu);
        index_min = (x < -epsilonk) | (x >= -epsilonk & x <= 0 & g >= mu);
        index_prox = (abs(x) <= epsilonk & abs(g) < mu) | (-epsilonk <= x & x < 0 & g <= -mu) | (0 < x & x <= epsilonk & g >= mu);
    
        I_plus = index_prox;
        I_minus = ~I_plus;
    
        w = zeros(n,1);
        w(index_max) = mu;
        w(index_min) = -mu;
        w(index_prox) = 0;
    
        gw = g+w;

        if ~useLBFGS
            gamma_vec = x - prox(x - g,mu);
            gamma_vec(I_minus) = gw(I_minus);
            gamma = 1e-4*norm(gamma_vec)^0.5;
        end

        if useLBFGS
            [pk,nPairs] = computePk_LBFGS( ...
                gw,I_plus,I_minus,S,Y,Hdiag,useCompiledLBFGS);
            out.lbfgs(k + 1) = nPairs;
            % The L-BFGS direction has no CG regularization parameter.
            linesearch_I_minus = (1-tao)*max(gw(I_minus)'*pk(I_minus),0);
        else
            Dx = p.*(1-p)/m;
            Aminus = A(:,I_minus);
            cgMaxIter = cgIterationLimit(res,out.gap(k + 1),opts);
            out.cgMaxIter(k + 1) = cgMaxIter;
            [pk,i,cg_Ax,cg_ATy] = computePk_CG( ...
                Aminus, Dx, gw, I_plus, I_minus, n, tao, gamma, ...
                useDiagPreconditioner,cgMaxIter,shouldStop);
            num_Ax = num_Ax + cg_Ax;
            num_ATy = num_ATy + cg_ATy;
            out.cg(k + 1) = i;
            linesearch_I_minus = (1-tao)*gamma*norm(pk(I_minus))^2;
        end

        % Restricting L-BFGS pairs to a changing working set can lose
        % positive curvature.  Restart with the scaled gradient if needed.
        if any(~isfinite(pk)) || gw'*pk <= 0
            pk = gw;
            if useLBFGS
                S = cell(0,1);
                Y = cell(0,1);
                Hdiag = 1;
                out.lbfgs(k + 1) = 0;
                linesearch_I_minus = (1-tao)*max(gw(I_minus)'*pk(I_minus),0);
            end
        end

        if ~isempty(shouldStop) && shouldStop()
            out.timedOut = true;
            break;
        end
        x = proj(xp-t*pk, mu, index_max, index_min, index_prox, t);
        
        nls = 0;
        while 1
            if ~isempty(shouldStop) && shouldStop()
                % Discard an unaccepted trial; keep the recorded iterate.
                x = xp;
                f = fp;
                out.timedOut = true;
                break;
            end
            Ax = A*x;
            num_Ax = num_Ax + 1;
            f = logisticState(A, b, Ax, x, mu, m);
    
            proxStep = (x(I_plus) - xp(I_plus))/t;
            if  fp - f >= sigma*t*linesearch_I_minus + sigma*t*(proxStep'*proxStep) %|| nls == 10
                break;
            end
                
            t = 0.2*t; nls = nls + 1;
            
            x = proj(xp-t*pk, mu, index_max, index_min, index_prox, t);
        end
        if out.timedOut; break; end
        out.nls(k + 1) = nls;
        
        out.dimension(k + 2) = nnz(x);
        
        g_old = g;
        if useLBFGS
            % The accepted objective is already available from line search.
            [g, p] = logisticGradient(A, b, Ax, m);
        else
            [~, g, p] = logisticState(A, b, Ax, x, mu, m);
        end
        num_ATy = num_ATy + 1;
        if useLBFGS
            [S,Y,Hdiag] = updateLBFGSHistory(S,Y,Hdiag,x-xp,g-g_old,opts.lbfgs_mem);
        end
    
        res = norm(x - prox(x - g,mu),2);
        out.res(k + 2) = res;
        if isempty(opts.f_ref)
            out.gap(k + 2) = NaN;
        else
            out.gap(k + 2) = relative_objective_gap(f, opts.f_ref);
        end
        
        epsilonk = min(epsilon, res);
        t = opts.alpha0;
    
        k = k + 1;
        out.fvec(k + 1) = f;
        out.num_Ax(k + 1) = num_Ax;
        out.num_ATy(k + 1) = num_ATy;
        out.mvp(k + 1) = num_Ax + num_ATy;
        out.timevec(k + 1) = toc(tt);
    
        % if opts.ftol > 0 && k > 8 && ...
        %         out.fvec(k - 7) - min(out.fvec(k - 6:k + 1)) < opts.ftol
        %     break
        % end
    end
    
    out.fvec = out.fvec(1:k + 1);
    out.dimension = out.dimension(1:k + 1);
    out.cg = out.cg(1:k);
    out.cgMaxIter = out.cgMaxIter(1:k);
    out.lbfgs = out.lbfgs(1:k);
    out.nls = out.nls(1:k);
    out.res = out.res(1:k + 1);
    out.gap = out.gap(1:k + 1);
    out.num_Ax = out.num_Ax(1:k + 1);
    out.num_ATy = out.num_ATy(1:k + 1);
    out.mvp = out.mvp(1:k + 1);
    out.timevec = out.timevec(1:k + 1);
    out.fval = f;
    out.itr = k;
    out.tt = toc(tt);
    out.runtime = out.tt;
    out.nrmG = res;
    out.option = option;
    out.lbfgs_mem = opts.lbfgs_mem;
    out.num_Ax_total = num_Ax;
    out.num_ATy_total = num_ATy;
    out.mvp_total = num_Ax + num_ATy;
end

    function maxIterations = cgIterationLimit(res,gap,opts)
    if isempty(opts.f_ref)
        nearTarget = res <= opts.cg_unlimited_factor*opts.gtol;
    else
        nearTarget = gap <= opts.cg_unlimited_factor*opts.gap_tol;
    end
    if ~opts.cg_adaptive || nearTarget
        maxIterations = Inf;
    elseif res >= 1e-1
        maxIterations = 10;
    elseif res >= 1e-2
        maxIterations = 15;
    elseif res >= 1e-3
        maxIterations = 20;
    else
        maxIterations = 25;
    end
    end

    function solver = normalizeSolver(value)
    if isnumeric(value)
        if ~isscalar(value) || ~(value == 1 || value == 2)
            error('The direction option must be 1 (L-BFGS) or 2 (CG).');
        end
        if value == 1; solver = 'lbfgs'; else; solver = 'cg'; end
        return;
    end

    if isstring(value); value = char(value); end
    if ~ischar(value)
        error('The direction option must be ''lbfgs'' or ''cg''.');
    end
    value = lower(strtrim(value));
    if any(strcmp(value, {'lbfgs','l-bfgs','quasi-newton','qn'}))
        solver = 'lbfgs';
    elseif any(strcmp(value, {'cg','conjugate-gradient','conjugate_gradient'}))
        solver = 'cg';
    else
        error('The direction option must be ''lbfgs'' or ''cg''.');
    end
    end

    function [S,Y,Hdiag] = updateLBFGSHistory(S,Y,Hdiag,s,y,maxPairs)
    ys = y'*s;
    yy = y'*y;
    if any(~isfinite(s)) || any(~isfinite(y)) || ...
            ~isfinite(ys) || ~isfinite(yy)
        return;
    end
    % Retain finite pairs and recheck curvature on the current working set.
    % Normalize the curvature test so it does not depend on gradient scale.
    curvatureScale = norm(s)*norm(y);
    if ys > 1e-10*curvatureScale && yy > 0
        Hdiag = ys/yy;
    end
    if numel(S) >= maxPairs
        S = S(2:end);
        Y = Y(2:end);
    end
    S{end+1,1} = s;
    Y{end+1,1} = y;
    end

    function [Pk,nPairs] = computePk_LBFGS( ...
            gw,I_plus,I_minus,S,Y,Hdiag,useCompiledLBFGS)
    Pk = zeros(size(gw));
    Pk(I_plus) = gw(I_plus);
    nPairs = 0;
    if ~any(I_minus); return; end

    % Reuse numeric indices across all history pairs on this working set.
    workingIndices = find(I_minus);
    valid = zeros(1,numel(S));
    for j = 1:numel(S)
        sj = S{j}(workingIndices);
        yj = Y{j}(workingIndices);
        ys = yj'*sj;
        curvatureScale = norm(sj)*norm(yj);
        if isfinite(ys) && ys > 1e-10*curvatureScale
            nPairs = nPairs + 1;
            valid(nPairs) = j;
        end
    end
    valid = valid(1:nPairs);
    if nPairs == 0
        Pk(I_minus) = Hdiag*gw(I_minus);
        return;
    end

    Swork = zeros(sum(I_minus),nPairs);
    Ywork = zeros(sum(I_minus),nPairs);
    for j = 1:nPairs
        Swork(:,j) = S{valid(j)}(workingIndices);
        Ywork(:,j) = Y{valid(j)}(workingIndices);
    end

    % PSSgb and TMP use the compiled L1General implementation on the
    % restricted working set.  Use the same tested kernel when available.
    if useCompiledLBFGS
        Pk(I_minus) = lbfgsC(gw(I_minus),Swork,Ywork,Hdiag);
    else
        Pk(I_minus) = lbfgsApply(gw(I_minus),S,Y,workingIndices,Hdiag,valid);
    end
    end

    function d = lbfgsApply(g,S,Y,workingIndices,Hdiag,valid)
    nPairs = numel(valid);
    q = g;
    alpha = zeros(nPairs,1);
    rho = zeros(nPairs,1);
    for ii = nPairs:-1:1
        j = valid(ii);
        sj = S{j}(workingIndices);
        yj = Y{j}(workingIndices);
        rho(ii) = 1/(yj'*sj);
        alpha(ii) = rho(ii)*(sj'*q);
        q = q-alpha(ii)*yj;
    end
    d = Hdiag*q;
    for ii = 1:nPairs
        j = valid(ii);
        sj = S{j}(workingIndices);
        yj = Y{j}(workingIndices);
        beta = rho(ii)*(yj'*d);
        d = d+sj*(alpha(ii)-beta);
    end
    end
    
    function y = prox(x, mu)
    y = max(abs(x) - mu, 0);
    y = sign(x) .* y;
    end

    function [f, g, p] = logisticState(A, b, Ax, x, mu, m)
    z = b .* Ax;
    f = sum(max(0, -z) + log1p(exp(-abs(z))))/m + mu*norm(x,1);
    if nargout > 1
        p = 1./(1 + exp(-z));
        g = A'*(b.*(p - 1))/m;
    end
    end

    function [g, p] = logisticGradient(A, b, Ax, m)
    z = b .* Ax;
    p = 1./(1 + exp(-z));
    g = A'*(b.*(p - 1))/m;
    end
    
    function y = proj(x, mu, index_max, index_min, index_prox, t)
    [n,~] = size(x);
    y = zeros(n,1);
    
    y(index_max) = max(x(index_max),0);
    y(index_min) = min(x(index_min),0);
    
    y(index_prox) = prox(x(index_prox),t*mu);
    
    end
    
    function [Pk,i,numAx,numATy] = computePk_CG( ...
            A, Dx, gw, I_plus, I_minus, n, tao, gamma, usePreconditioner,maxIterations,shouldStop)
    Pk = zeros(n,1);
    numAx = 0;
    numATy = 0;
    if ~any(I_minus)
        Pk(I_plus) = gw(I_plus);
        i = 0;
    elseif all(I_minus)
        [Pk,i,numAx,numATy] = conjgrad( ...
            A, Dx, gw(I_minus), zeros(sum(I_minus),1), tao, gamma, ...
            usePreconditioner,maxIterations,shouldStop);
    else
        [Pk(I_minus),i,numAx,numATy] = conjgrad( ...
            A, Dx, gw(I_minus), zeros(sum(I_minus),1), tao, gamma, ...
            usePreconditioner,maxIterations,shouldStop);
        Pk(I_plus) = gw(I_plus);
    end
    end
    
    function [x,i,numAx,numATy] = conjgrad( ...
            A, Dx, b, x, tao, gamma, usePreconditioner,maxIterations,shouldStop)
        numAx = 0;
        numATy = 0;
        Ax = A*x;
        numAx = numAx + 1;
        r = b - A'*(Dx.*(Ax)) - gamma*x;
        numATy = numATy + 1;
        if usePreconditioner
            % diag(A' * diag(Dx) * A + gamma * I)
            Mdiag = full((A.^2)'*Dx) + gamma;
            Mdiag = max(Mdiag,eps);
            z = r./Mdiag;
        else
            z = r;
        end
        p = z;
        rzold = r' * z;
        if rzold == 0
            i = 0;
            return;
        end
        normb = norm(b);
        i = 0;
        for cgStep = 1:min(length(b),maxIterations)
            if ~isempty(shouldStop) && shouldStop()
                break;
            end
            i = cgStep;
            Ap = A * p;
            numAx = numAx + 1;
            Hp = A'*(Dx.*(Ap))+gamma*p;
            numATy = numATy + 1;
            pHp = p' * Hp;
            if pHp <= 0
                break;
            end
            alpha = rzold / pHp;
            x = x + alpha * p;
            r = r - alpha * Hp;
            if usePreconditioner
                z = r./Mdiag;
            else
                z = r;
            end
            rznew = r' * z;
            % if sqrt(rsnew) < 1e-10
            %     break;
            % end
            if norm(r) < min(tao*min(normb,gamma*norm(x)), 1e-3)
                break;
            end
            p = z + (rznew / rzold) * p;
            rzold = rznew;
        end
    end
