function [x,it,info,r,rnm] = mypcg_pssgb(Afun,b,rtol,maxit,precf,sig)
%MYPCG_PSSGB Truncated PCG for (H + sig*I)x = b.
% Afun applies H to a vector. The stopping test matches the one used by
% TMAP: ||r|| <= rtol*min(||b||,sig*||x||).

if nargin < 5 || isempty(precf)
    precavail = false;
else
    precavail = true;
end
if nargin < 6 || ~isscalar(sig) || ~isfinite(sig) || sig <= 0
    error('sig must be a positive finite scalar.');
end

x = zeros(size(b));
r = b;
bnm = norm(b);
rnm = bnm;
info = 1;
it = 0;

if ~isfinite(rnm) || rnm > 1e16
    info = 4;
    return;
end
if bnm == 0
    return;
end

if precavail
    z = precf(r);
else
    z = r;
end
p = z;
rho = r'*z;

if ~isfinite(rho) || rho <= 0
    info = 2;
    return;
end

for it = 1:maxit
    q = Afun(p) + sig*p;
    ptq = p'*q;
    if ~isfinite(ptq) || ptq <= 0
        info = 2;
        return;
    end

    alpha = rho/ptq;
    x = x + alpha*p;
    r = r - alpha*q;
    rnm = norm(r);

    if rnm <= rtol*min(bnm,sig*norm(x))
        return;
    end

    if precavail
        z = precf(r);
    else
        z = r;
    end
    rhoNew = r'*z;
    if ~isfinite(rhoNew) || rhoNew <= 0
        info = 2;
        return;
    end
    p = z + (rhoNew/rho)*p;
    rho = rhoNew;
end

info = 3;
end
