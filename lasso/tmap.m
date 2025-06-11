function [x, out] = tmap(A,At,b,N,mu,opts)

tic;
%%-------------------------------------------------------------------------
if nargin < 5; opts = []; end
if ~isfield(opts,'x0');    opts.x0  = zeros(N,1);   end
if ~isfield(opts,'tol');   opts.tol = 1e-6;     end
if ~isfield(opts,'record');opts.record = 1;     end
if ~isfield(opts,'eps');   opts.eps = 1e-4;     end

if ~isfield(opts,'comp_obj'); opts.comp_obj = 0;   end
if ~isfield(opts,'tau');      opts.tau = 6;   end
if ~isfield(opts,'tau_adapt');opts.tau_adapt = 0;   end
if ~isfield(opts,'tau_m');    opts.tau_m = 1e-3;   end
if ~isfield(opts,'tau_M');    opts.tau_M = 1e+4;   end
if ~isfield(opts,'maxitTau'); opts.maxitTau = 20;   end
if ~isfield(opts,'teta');     opts.teta = 0.85;   end
if ~isfield(opts,'beta');     opts.beta = 1e-1;   end
if ~isfield(opts,'gamma');    opts.gamma = 1e-1;   end
if ~isfield(opts,'CG_maxit'); opts.CG_maxit = 5;   end
if ~isfield(opts,'CG_tol');   opts.CG_tol = 1e-1;   end
if ~isfield(opts,'CG_adapt'); opts.CG_adapt = 1;   end
if ~isfield(opts,'cont');     opts.cont = 1;   end
if ~isfield(opts,'cont_max'); opts.cont_max = 10;   end
if ~isfield(opts,'nnewtcomp'); opts.nnewtcomp = 10;   end

if ~isfield(opts,'crit'); opts.crit = 1;   end

x      = opts.x0;
tol    = opts.tol;
record = opts.record;
eps    = opts.eps;
tau    = opts.tau; 
nnewtcomp = opts.nnewtcomp;
resnp = zeros(nnewtcomp,1);
rnpinx = 0;

if opts.crit == 2
    fopt = opts.fopt;
end

%     if opts.tol >= 1e-5; opts.CG_maxit = 3;
%     else opts.CG_maxit = 8; end



if opts.tau_adapt
    maxitTau = opts.maxitTau;
    tau_m    = opts.tau_m;
    tau_M    = opts.tau_M;
    teta     = opts.teta;
    Qt       = 1;
end

CG_tol = opts.CG_tol;
CG_maxit = opts.CG_maxit;

%%-------------------------------------------------------------------------
% options for the Levenberg-Marquardt algorithm
if ~isfield(opts,'maxit');    opts.maxit = 100;   end
if ~isfield(opts,'xtol');     opts.xtol = 1e-6;   end
if ~isfield(opts,'ftol');     opts.ftol = 1e-12;  end
if ~isfield(opts,'muPow');    opts.muPow = 0.5;     end
if ~isfield(opts,'resFac');   opts.resFac = 0.98;   end
if ~isfield(opts,'eta1');     opts.eta1 = 1e-6;   end
if ~isfield(opts,'eta2');     opts.eta2 = 0.9;    end
if ~isfield(opts,'gamma1');   opts.gamma1 = 0.5;  end
if ~isfield(opts,'gamma2');   opts.gamma2 = 1;    end
if ~isfield(opts,'gamma3');   opts.gamma3 = 10;   end
if ~isfield(opts,'lambda');   opts.lambda = 0.1;    end %adjust mu
if ~isfield(opts,'itPrint');  opts.itPrint = 1;   end
if ~isfield(opts,'maxItStag');opts.maxItStag = 5; end
if ~isfield(opts,'restart');  opts.restart = 1; end
if ~isfield(opts,'maxItPCG'); opts.maxItPCG = 20;   end

maxItStag = opts.maxItStag; restart=opts.restart; maxItPCG=opts.maxItPCG;
maxit   = opts.maxit;   itPrint = opts.itPrint; muPow   = opts.muPow;
xtol    = opts.xtol;    ftol    = opts.ftol;    resFac  = opts.resFac;
eta1    = opts.eta1;    eta2    = opts.eta2;    lambda  = opts.lambda;
epsilon = 1e-6; sigma = 1e-1; tao = 1e-1;
%-------------------------------------------------------------------------
out = struct();
%------------------------------------------------------------------

% continuation
muf     = mu;

if opts.cont
    bsup        = max(abs(b));
    mu0         = min(0.25,2.2*(bsup/muf)^(-1/3))*bsup;
    mu          = max(muf,mu0);
    
    cont_max    = opts.cont_max;
    cont_count  = 1;
end

% tolerances
if opts.CG_adapt
    res_tol = 1e-3;
end
res         = 1e20;
ftol        = 1e-4;

% first computations
Atb         = At(b);
btb         = b'*b;

% compute residual function p = F_tau(x)
[~, res, ~, grad] = comp_res(x);
xp = x;  resp = res; 
epsilonk = min(epsilon,res);

if opts.cont; resp_mu = res; end

% stage two: Levenberg-Marquardt algorithm
% itStag = 0; iterd = 1; flag = []; relres = [];

nr_CG = 0;
nr_res = 1;
nr_Acall = 3;

itTau = 0;
switch_pow = 1;

funcx = func(x);

for iter = 1:maxit
    itTau = itTau + 1;

    %----------------------------------------------------------------------
    % parameter for the revised Newton system
    % if switch_pow
    %     if res/tau < 1e-9
    %         muPow = 0.15;   %60dB 0.1 20dB 0.15
    %     elseif res/tau < 1e-3
    %         muPow = 0.6;
    %     elseif res/tau < 1e-2
    %         muPow = 0.6;
    %     elseif res/tau < 10/6
    %         muPow = 0.7;
    %     else
    %         muPow = 0.7;
    %     end
    % end
    muPow = 0.7;

    if switch_pow
        if res < 0.1
            lambda = 0.1;
        elseif res < 50
            lambda = 0.1;
        else
            lambda = 0.1;
        end
    end

    
    sig = lambda*(resp^muPow);
    %sig = lambda*min(1,resp)^muPow;
    
    index_max = (x > epsilonk) | (0 <= x & x <= epsilonk & grad <= -mu);
    index_min = (x < -epsilonk) | (x >= -epsilonk & x <= 0 & grad >= mu);
    I_minus = logical(index_max + index_min);
    I_plus = ~I_minus;

    w = zeros(N,1);
    w(index_max) = mu;
    w(index_min) = -mu;

    gw = grad + w;

    % reduce the Newton system and solve it via CG
    d   = grad;

    Mat         = @(y)build_CGMatrix(I_plus,y);
        
    [d(I_minus),iterd,~,~,~,~,~] = ...
        mypcgw(Mat,gw(I_minus),CG_tol,CG_maxit,[],1);
    
    nr_CG = nr_CG + iterd;
    nr_Acall = nr_Acall + iterd*2;
    %----------------------------------------------------------------------
    
    % line search
    t = 1;
    [xz, nrm_dx_plus] = proj(x, d, t);

    linesearch_I_minus = (1-tao)*sig*norm(d(I_minus))^2;
    
    nls = 0;
    while 1
        funcxz = func(xz);
        nr_Acall = nr_Acall + 1;
        if  funcxz - funcx <= -sigma*t*linesearch_I_minus - sigma/t*nrm_dx_plus^2 || nls == 5
            break;
        end
            
        t = 0.2*t; nls = nls + 1;
        [xz, nrm_dx_plus] = proj(x, d, t);
    end

    [~, resxz, ~, gradxz] = comp_res(xz);
    nr_Acall = nr_Acall + 2;
    x = xz; res = resxz; grad = gradxz; epsilonk = min(epsilon, res);
    funcx = funcxz; fiter = funcxz;
             
    % stopping criteria
    switch opts.crit
        case 1
            cstop = (mu <= (1+1e-10)*muf) && (res <= tol);
        case 2
            if mu <= (1+1e-10)*muf 
                % fiter = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);
                cstop = ((fiter - fopt)/max(1,abs(fopt)) <= tol);
            else
                cstop = 0;
            end 
    end    
    
    if cstop
        if opts.crit == 2
            out.f = fiter;
        end
        out.msg = 'optimal';
        break;
    end
       
        % evaluate the quality of the last iterate
        if opts.cont
            goodIter = (cont_count >= cont_max) ||  ...
             ((res <= 0.5*resp) || (res <= max(0.1*resp_mu, ftol)));
        else
            goodIter = (res <= 0.5*resp)|| (res <= max(0.1*resp_mu, ftol));
        end
            
        % continuation: update mu
        if opts.cont
            if goodIter
                mup         = mu;
                muredf      = max((1-(0.65^log10(mu0/mu))*0.535),0.15);
                mu          = max(muf,muredf*mu);
                                
                if mu > muf
                    cont_count      = 1;
                else
                    cont_count      = cont_count + 1;
                end
            else
                cont_count = cont_count + 1;
            end
        end
        
        resp_mu = res; 

     %   update CG parameters if final mu is reached
        if opts.CG_adapt && (mu <= (1+1e-10)*muf)
            if res < res_tol
                CG_tol     = max(0.08*CG_tol,1e-7);
                res_tol    = 0.1*res_tol;
            end
            
            if res < 1e-3
                CG_maxit   = min(max(CG_maxit + 7,3),300);
            elseif res < 1e-2
                CG_maxit   = 10;
            elseif res < 1e-1
                CG_maxit   = 6;
            else
                CG_maxit   = 5;
            end
        end

        resp = res;
    
end %end outer loop
out.time = toc;

out.iter = iter;
out.res  = res;
out.f = 0.5*((grad-Atb)'*x + btb) + mu*norm(x,1);

out.nr_res = nr_res;
out.nr_CG = nr_CG;
out.Acalls = nr_Acall;% + nr_res*2 + nr_CG*2;

    function a = idot(x,y)
        a = real(x(:)'*y(:));
    end

    % computes a reduced generalized derivative for the CG method
    function [z,BtBr] = build_CGMatrix(ind,y)
        z           = zeros(N,1);
        z(~ind)     = y;
        BtBr        = At(A(z));
        z           = BtBr(~ind) + (sig/tau)*y;
        %z           = BtBr(~ind) + sig*y;
    end

    function ss = shringkage(xx,mumu)
        ss = sign(xx).*max(abs(xx)-mumu,0);
    end

%     function [Ftx, res, yx, grad, obj] = comp_res(xx)
%         grad   = At(A(xx)) - Atb;
%         yx     = xx - tau*grad;
%         Ftx    = xx - shringkage(yx,mu*tau);
%         %res    = norm(Ftx); 
%         res    = norm(Ftx)/tau;        
%         obj    = mu*norm(x,1) + 0.5*res^2;
%     end
% 
%     function [Ftx, res, yx, obj] = comp_resR(xx,grad)
%         yx     = xx - tau*grad;
%         Ftx    = xx - shringkage(yx,mu*tau);
%         %res    = norm(Ftx);
%         res    = norm(Ftx)/tau;
%         obj    = mu*norm(x,1) + 0.5*res^2;
%     end

    function [Ftx, res, yx, grad] = comp_res(xx)
        grad   = At(A(xx)) - Atb;
        yx     = xx - grad;
        Ftx    = xx - shringkage(yx,mu);
        res    = norm(Ftx); 
    end

    function fv = func(xx)
       fv = mu*norm(xx,1) + 0.5*norm(A(xx)-b)^2; 
    end

    function [Ftx, res, yx] = comp_resR(xx,grad)
        yx     = xx - grad;
        Ftx    = xx - shringkage(yx,mu);
        res    = norm(Ftx);
    end

    function [y, nrm_dx_plus] = proj(xx, d, t)
        z = xx - t*d;
        
        y = shringkage(z,t*mu);

        y(index_max) = 0;
        y(index_min) = 0;
        xx(index_max) = 0;
        xx(index_min) = 0;
        dx_plus = y - xx;
        nrm_dx_plus = norm(dx_plus);

        y(index_max) = max(z(index_max),0);
        y(index_min) = min(z(index_min),0);

        
    end


end



