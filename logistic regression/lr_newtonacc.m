function [x, out] = lr_newtonacc(x0, A, b, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'f_ref'); opts.f_ref = []; end
if ~isfield(opts, 'gap_tol'); opts.gap_tol = []; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 0; end
if ~isfield(opts, 'opt_dim'); opts.opt_dim = 0; end

out = struct();
[m,n] = size(A);
k = 0;
x = x0;
t = opts.alpha0;
fp = inf;

tt = tic;
mvp = 2; % initial A*x and A'*gradient products

Ax = A*x;
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,1);
g = A'*(b./(1+expba) - b)/m;
tmpf = f;
y = x;

res = norm(x - prox(x - g,mu),2);
out.fvec = [];
out.dimension = [];
out.res = res;
out.timevec = toc(tt);
out.mvpvec = mvp;
out.gap = relative_objective_gap(f, opts.f_ref);


Cval = tmpf; Q = 1; gamma = 0.85; rhols = 1e-6; m1 = 0.1; rate = 1;

while k < opts.maxit && ((isempty(opts.f_ref) && res > opts.gtol) || ...
        (~isempty(opts.f_ref) && out.gap(end) > opts.gap_tol))
    
    fp = f;
    xp = y;
    
    x = prox(xp - t * g, t * mu);
    
    if opts.ls
        nls = 0;
        while 1
            Ax = A*x;
            mvp = mvp + 1;
            expba = exp(- b.*Ax);
            tmpf = sum(log(1 + expba))/m + mu*norm(x,1);
            if tmpf <= Cval - rhols*0.5*t*norm(x-xp,2)^2 || nls == 10
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(xp - t * g, t * mu);
        end
        
        f = tmpf;
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmpf)/Q;
    
    else
        Ax = A*x;
        expba = exp(- b.*Ax);
        f = sum(log(1 + expba))/m + mu*norm(x,1);
    end
    
    g = A'*(b./(1+expba) - b)/m;
    mvp = mvp + 1;
    gp = g;
    Fx = f;
    dim = sum(abs(x)>0);
    out.dimension = [out.dimension, dim];

    manifold_index = (abs(x) > 0);
    r_grad = egrad_to_rgrad(x, manifold_index, g, mu);
    p = 1./(1 + expba);
    Dx = p.*(1-p)/m;
    Aminus = A(:,manifold_index);
    eta = zeros(n,1);
    [eta(manifold_index),~,cg_mvp] = conjgrad(Aminus, Dx, -r_grad(manifold_index), zeros(sum(manifold_index),1), rate);
    mvp = mvp + cg_mvp;

    alpha = 1;
    linesearch = eta'*r_grad;
    if linesearch > 0 
        eta = -eta;
    end
    Aeta = Aminus*eta(manifold_index);
    mvp = mvp + 1;
    y = x + alpha*eta;
    nls = 0;
    while 1
        Ay = Ax + alpha*Aeta;
        expba = exp(- b.*Ay);
        tmpf = sum(log(1 + expba))/m + mu*norm(y,1);
        if tmpf <= Fx + m1*alpha*linesearch || nls == 5
            break;
        end
        alpha = 0.2*alpha; nls = nls + 1;
        y = x + alpha*eta;
    end
    f = tmpf;
    if alpha == 1
        rate = 0.1* rate;
    end

    g = A'*(b./(1+expba) - b)/m;
    mvp = mvp + 1;
    res = norm(y - prox(y - g,mu),2);
    out.res = [out.res, res];
    out.timevec = [out.timevec, toc(tt)];
    out.mvpvec = [out.mvpvec, mvp];
    out.gap = [out.gap, relative_objective_gap(f, opts.f_ref)];

    if opts.bb && opts.ls
        dx = y - x;
        dg = g - gp;
        dxg = abs(dx'*dg);
        
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,2)^2/dxg;
            else
                t = dxg/norm(dg,2)^2;
            end
        end
        
        t = min(max(t,opts.alpha0),1e12);

    else
        t = opts.alpha0;
    end
    
    k = k + 1;
    out.fvec = [out.fvec, f];
    if isempty(opts.f_ref) && k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break;
    end
end


out.fvec = out.fvec(1:k);
out.dimension = out.dimension(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = res;
out.mvp_total = mvp;
end
%% 辅助函数
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end

function [y] = Manifoldupdate_TNCG(x, e_grad, Fx, A, b, mu)
manifold_index = (abs(x) > 0);
[n,~] = size(e_grad);
r_grad = egrad_to_rgrad(x, manifold_index, e_grad, mu);
[Aminus, Dx] = ehess_to_rhess(manifold_index, A, b, x);
eta = zeros(n,1);
eta(manifold_index) = TNCG(Aminus, Dx, -r_grad(manifold_index), zeros(sum(manifold_index),1));
y = armijo(x, eta, r_grad, Fx, A, b, mu);
end

function [y] = Manifoldupdate_newton(x, e_grad, Fx, A, b, mu)
manifold_index = (abs(x) > 0);
[n,~] = size(e_grad);
r_grad = egrad_to_rgrad(x, manifold_index, e_grad, mu);
[Aminus, Dx] = ehess_to_rhess(manifold_index, A, b, x);
eta = zeros(n,1);
eta(manifold_index) = conjgrad(Aminus, Dx, -r_grad(manifold_index), zeros(sum(manifold_index),1));
y = armijo(x, eta, r_grad, Fx, A, b, mu);
end

function rgrad = egrad_to_rgrad(x, manifold_index, e_grad, mu)
[n,~] = size(e_grad);
rgrad = zeros(n,1);
rgrad(manifold_index) = e_grad(manifold_index);
rgrad = rgrad + mu*sign(x);
end

function [Aminus, Dx] = ehess_to_rhess(manifold_index, A, b, x)
[m,~] = size(A);
Ax = A*x;
expba = exp(- b.*Ax);
p = 1./(1 + expba);
Dx = p.*(1-p)/m;
Aminus = A(:,manifold_index);
end

function y = retraction(x, eta)
y = x + eta;
end

function [y, alpha] = armijo(x, eta, r_grad, Fx, A, b, mu)
m1 = 1e-1;
alpha = 1;
linesearch = eta'*r_grad;
if linesearch > 0 
    eta = -eta;
end
y = retraction(x, alpha*eta);
nls = 0;
[m,~] = size(A);
while 1
    Ay = A*y;
    expba = exp(- b.*Ay);
    tmpf = sum(log(1 + expba))/m + mu*norm(y,1);
    if tmpf <= Fx + m1*alpha*linesearch || nls == 10
        break;
    end
    alpha = 0.2*alpha; nls = nls + 1;
    y = retraction(x, alpha*eta);
end
if nls == 5
    y = x;
end
end

function y = sign(x)
plus = x >0;
minus = x<0;
[n,~] = size(x);
y = zeros(n,1);
y(plus) = 1;
y(minus) = -1;
end

function [x, i, mvp] = conjgrad(A, Dx, b, x, rate)
    mvp = 2;
    Ax = A*x;
    r = b - A'*(Dx.*(Ax));
    p = r;
    rsold = r' * r;
    temp = norm(b);

    for i = 1:50
        Ap = A * p;
        mvp = mvp + 1;
        Hp = A'*(Dx.*(Ap));
        mvp = mvp + 1;
        alpha = rsold / (p' * Hp);
        x = x + alpha * p;
        r = r - alpha * Hp;
        rsnew = r' * r;
        if abs((r'-b')* x)/(x'*x) < rate
            break;
        end
        if sqrt(rsnew) < temp^1.5
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
