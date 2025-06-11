function [x, out] = lr_2mproj(x0, A, b, mu, opts, option)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
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

Ax = A*x;
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,1);
g = A'*(b./(1+expba) - b)/m;
p = 1./(1 + expba);

res = norm(x - prox(x - g,mu),2);
out.fvec = [];
out.dimension = [];
out.cg = [];


sigma = 1e-1;
epsilon = 1e-3;
lambda = 1e-4;
tao = 1e-1;
delta = res;
epsilonk = min(epsilon,delta);
if_opt_dim = 1;

while k < opts.maxit && res > opts.gtol %&& abs(f - fp) > opts.ftol
    
    fp = f;
    xp = x;
    gp = g;

    index_max = (x > epsilonk) | (0 <= x & x <= epsilonk & g <= -mu);
    index_min = (x < -epsilonk) | (x >= -epsilonk & x <= 0 & g >= mu);
    index_prox = (abs(x) <= epsilonk & abs(g) < mu) | (-epsilonk <= x & x < 0 & g <= -mu) | (0 < x & x <= epsilonk & g >= mu);

    I_plus = index_prox;
    I_minus = xor(ones(n,1),I_plus);

    w = zeros(n,1);
    w(index_max) = mu;
    w(index_min) = -mu;
    w(index_prox) = 0;

    Dx = p.*(1-p)/m;
    Aminus = A(:,I_minus);

    gw = g+w;

        gamma_vec = x - prox(x - g,mu);

        gamma_vec(I_minus) = gw(I_minus);

        gamma = lambda*norm(gamma_vec)^0.5;

    [pk,i] = computePk_CG(Aminus, Dx, gw, I_plus, I_minus, n, tao, gamma);
    out.cg = [out.cg, i];

    linesearch_I_minus = (1-tao)*gamma*norm(pk(I_minus))^2;

    x = proj(xp-t*pk, mu, index_max, index_min, index_prox, t);
    
    nls = 0;
    while 1
        Ax = A*x;
        expba = exp(- b.*Ax);
        f = sum(log(1 + expba))/m + mu*norm(x,1);

        if  fp - f >= sigma*t*linesearch_I_minus + sigma/t*norm(x(I_plus)-xp(I_plus),2)^2 || nls == 10
            break;
        end
            
        t = 0.2*t; nls = nls + 1;
        
        x = proj(xp-t*pk, mu, index_max, index_min, index_prox, t);
    end
    
    dim = sum(abs(x)>0);
    out.dimension = [out.dimension, dim];
    if dim == opts.opt_dim && if_opt_dim
        out.t_id = toc(tt);
        out.itr_id = k;
        if_opt_dim = 0;
    end
    
    g = A'*(b./(1+expba) - b)/m;
    p = 1./(1 + expba);

    s = x-xp;
    y = g-gp;

    res = norm(x - prox(x - g,mu),2);
    
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

function y = proj(x, mu, index_max, index_min, index_prox, t)
[n,~] = size(x);
y = zeros(n,1);

y(index_max) = max(x(index_max),0);
y(index_min) = min(x(index_min),0);

y(index_prox) = prox(x(index_prox),t*mu);

end

function [Pk,i] = computePk_CG(A, Dx, gw, I_plus, I_minus, n, tao, gamma)
Pk = zeros(n,1);
if all(I_minus)
    [Pk,i] = conjgrad(A, Dx, gw(I_minus), zeros(sum(I_minus),1), tao, gamma);
else
    Pk = gw;
    [Pk(I_minus),i] = conjgrad(A, Dx, gw(I_minus), zeros(sum(I_minus),1), tao, gamma);
end
end

function [x,i] = conjgrad(A, Dx, b, x, tao, gamma)
    Ax = A*x;
    r = b - A'*(Dx.*(Ax)) - gamma*x;
    p = r;
    rsold = r' * r;
    for i = 1:length(b)
        Ap = A * p;
        Hp = A'*(Dx.*(Ap))+gamma*p;
        alpha = rsold / (p' * Hp);
        x = x + alpha * p;
        r = r - alpha * Hp;
        rsnew = r' * r;
        % if sqrt(rsnew) < 1e-10
        %     break;
        % end
        if sqrt(rsnew) < tao*min(1e-3,gamma*norm(x))
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

function pk = computePk_lbfgs(old_dirs, old_stps, b, I_plus, I_minus, Hdiag, n)
pk = zeros(n,1);

curvSat = sum(old_dirs(I_minus,:).*old_stps(I_minus,:)) > 1e-10;
pk(I_minus) = lbfgs(b(I_minus),old_dirs(I_minus,curvSat),old_stps(I_minus,curvSat),Hdiag);
pk(I_plus) = b(I_plus);
end