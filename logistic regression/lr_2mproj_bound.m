function [x, out] = lr_2mproj_bound(x0, A, b, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 0; end

out = struct();
[m,n] = size(A);
mu1 = mu*ones(n,1);
k = 0;

x = x0;
t = opts.alpha0;
fp = inf;

tt = tic;

Ax = A*x;
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,1);
g = A'*(b./(1+expba) - b)/m;
p = 1./(1 + expba);


out.fvec = f;

x_plus = max(x, 0);
x_minus = -min(x, 0);
res = norm([x_plus-max(x_plus - (g+mu),0);x_minus-max(x_minus - (-g+mu),0)] ,2);
out.dimension= sum(x_plus>0)+sum(x_minus>0);
out.cg = [];
out.res = res;


sigma = 1e-1;
epsilon = 1e-3;
delta = res;
epsilonk = min(epsilon,delta);

while k < opts.maxit && res > opts.gtol %&& abs(f - fp) > opts.ftol

    
    fp = f;
    gp = g;
    xp_plus = x_plus;
    xp_minus = x_minus;

    index_x_plus_plus = (0 <= x_plus & x_plus <= epsilonk & g > -mu);
    index_x_minus_plus = (0 <= x_minus & x_minus <= epsilonk & g < mu);
    index_x_plus_minus = xor(ones(n,1), index_x_plus_plus);
    index_x_minus_minus = xor(ones(n,1), index_x_minus_plus);

    if any(index_x_plus_minus & index_x_minus_minus)
        fprintf('%d error\n',k);
    end

    % w = zeros(n,1);
    % w(index_max) = mu;
    % w(index_min) = -mu;
    % w(index_prox) = 0;

    Dx = p.*(1-p)/m;

    % if k==0
    %     pk_plus = g+mu;
    %     pk_minus = -g+mu;
    %     old_dirs = zeros(2*n,0);
    %     old_stps = zeros(2*n,0);
    %     Hdiag = 1;
    % else
    %     corrections = 100;
    %     numCorrections = size(old_dirs,2);
    %         if numCorrections < corrections
    %             % Full Update
    %             old_dirs(:,numCorrections+1) = s;
    %             old_stps(:,numCorrections+1) = y;
    %         else
    %             % Limited-Memory Update
    %             old_dirs = [old_dirs(:,2:corrections) s];
    %             old_stps = [old_stps(:,2:corrections) y];
    %         end
    % 
    %         % Update scale of initial Hessian approximation
    %         ys = y'*s;
    %         if ys > 1e-10
    %             Hdiag = ys/(y'*y);
    %         end
    % 
    %         [pk_plus, pk_minus] = computePk_lbfgs(old_dirs, old_stps, g+mu, -g+mu, index_x_plus_plus, index_x_plus_minus, index_x_minus_plus, index_x_minus_minus, Hdiag, n);
    % 
    % % [pk_plus, pk_minus, B] = computePk_bfgs(B, y, s, g+mu, -g+mu, index_x_plus_plus, index_x_plus_minus, index_x_minus_plus, index_x_minus_minus, n);
    % end

    A_x_plus_minus = A(:,index_x_plus_minus);
    A_x_minus_minus = A(:,index_x_minus_minus);
    [pk_plus, pk_minus, i] = computePk_CG(A_x_plus_minus, A_x_minus_minus, Dx, g+mu, -g+mu, index_x_plus_plus, index_x_plus_minus, index_x_minus_plus, index_x_minus_minus, n);
    out.cg = [out.cg, i];

    if all(~index_x_plus_minus)
            gtpk_I_minus = (- g(index_x_minus_minus)+mu1(index_x_minus_minus))'*pk_minus(index_x_minus_minus) ;
    elseif all(~index_x_minus_minus)
            gtpk_I_minus = (g(index_x_plus_minus)+mu1(index_x_plus_minus))'*pk_plus(index_x_plus_minus);
    else
        gtpk_I_minus = (- g(index_x_minus_minus)+mu1(index_x_minus_minus))'*pk_minus(index_x_minus_minus)+(g(index_x_plus_minus)+mu1(index_x_plus_minus))'*pk_plus(index_x_plus_minus);
    end

    [x_plus, x_minus] = proj(xp_plus, xp_minus, pk_plus, pk_minus, t);
    x = x_plus - x_minus;
    
    nls = 0;
    while 1
        Ax = A*x;
        expba = exp(- b.*Ax);
        f = sum(log(1 + expba))/m + mu*norm(x,1);

        if all(~index_x_plus_plus)
            gt_deltax = (-g(index_x_minus_plus)+mu)'*(-x_minus(index_x_minus_plus)+xp_minus(index_x_minus_plus));
        elseif all(~index_x_minus_plus)
            gt_deltax = (g(index_x_plus_plus)+mu)'*(-x_plus(index_x_plus_plus)+xp_plus(index_x_plus_plus));
        else
            gt_deltax = (-g(index_x_minus_plus)+mu)'*(-x_minus(index_x_minus_plus)+xp_minus(index_x_minus_plus)) + (g(index_x_plus_plus)+mu)'*(-x_plus(index_x_plus_plus)+xp_plus(index_x_plus_plus));
        end

        

        if  (fp - f >= sigma*t*gtpk_I_minus + sigma*gt_deltax) || nls == 10
            break;
        end
            
        t = 0.2*t; nls = nls + 1;
        
        [x_plus, x_minus] = proj(xp_plus, xp_minus, pk_plus, pk_minus, t);
        x = x_plus - x_minus;
    end
    % fprintf('%10d %10d %15.5e\n',k,t,f);

    out.dimension = [out.dimension, sum(x_plus>0)+sum(x_minus>0)];
    
    g = A'*(b./(1+expba) - b)/m;
    p = 1./(1 + expba);
    
    s = [x_plus-xp_plus; x_minus-xp_minus];
    y = [g-gp; gp-g];

    res = norm([x_plus-max(x_plus - (g+mu),0);x_minus-max(x_minus - (-g+mu),0)] ,2);
    out.res = [out.res, res];
    
    epsilonk = min(epsilon, res);
    t = opts.alpha0;

    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, res);
    end

    if k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break
    end
end

out.fvec = out.fvec(1:k);
out.dimension = out.dimension(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = res;

end

function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end

function [y1, y2] = proj(x_plus, x_minus, pk_plus, pk_minus, t)
y1 = max(x_plus-t*pk_plus,0);
y2 = max(x_minus-t*pk_minus,0);
end

function y = sign(x)
[n,~] = size(x);
y = zeros(n,1);

y(x>0) = 1;
y(x<0) = -1;
y(x==0) = 0;

end

function [pk_plus, pk_minus, i] = computePk_CG(A1, A2, Dx, b1, b2, index_x_plus_plus, index_x_plus_minus, index_x_minus_plus, index_x_minus_minus, n)
pk_plus = zeros(n,1);
pk_minus = zeros(n,1);
if all(~index_x_plus_minus) && all(~index_x_minus_minus)
    pk_plus = b1;
    pk_minus = b2;
    i=0;
elseif all(~index_x_plus_minus)
    pk_plus = b1;
    [pk_minus(index_x_minus_minus), i] = conjgrad1(A2, Dx, b2(index_x_minus_minus));
elseif all(~index_x_minus_minus)
    pk_minus = b2;
    [pk_plus(index_x_plus_minus), i] = conjgrad1(A1, Dx, b1(index_x_plus_minus));
else
    [pk_plus(index_x_plus_minus), pk_minus(index_x_minus_minus), i] = conjgrad(A1, A2, Dx, b1(index_x_plus_minus), b2(index_x_minus_minus));
    pk_plus(index_x_plus_plus) = b1(index_x_plus_plus);
    pk_minus(index_x_minus_plus) = b2(index_x_minus_plus);
end
end

function [x,y,i] = conjgrad(A1, A2, Dx, b1, b2)
    b = [b1; b2];
    % A = [A1 A2];
    x = zeros(length(b1),1);
    y = zeros(length(b2),1);
    A1x = A1*x;
    A2y = A2*y;
    r = b - [A1'*(Dx.*(A1x-A2y)); A2'*(Dx.*(-A1x+A2y))];
    p = r;
    rsold = r' * r;
    temp = norm(b);
    gamma = min(1e-6 , temp);
    for i = 1:length(b)
        A1p = A1*p(1:length(b1));
        A2p = A2*p(1+length(b1):length(b2)+length(b1));
        Hp = [A1'*(Dx.*(A1p-A2p)); A2'*(Dx.*(-A1p+A2p))];
        alpha = rsold / (p' * Hp);
        x = x + alpha * p(1:length(b1));
        y = y + alpha * p(1+length(b1):length(b2)+length(b1));
        r = r - alpha * Hp;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
            break;
        end
        if sqrt(rsnew) < min(temp^1.5,0.1*temp)
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

function [x,i] = conjgrad1(A, Dx, b)
x = zeros(length(b),1);
        Ax = A*x;
        r = b - A'*(Dx.*(Ax));
        p = r;
        rsold = r' * r;
        for i = 1:length(b)
            Ap = A * p;
            Hp = A'*(Dx.*(Ap));
            alpha = rsold / (p' * Hp);
            x = x + alpha * p;
            r = r - alpha * Hp;
            rsnew = r' * r;
            if sqrt(rsnew) < 1e-10
                break;
            end
            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        end
    end