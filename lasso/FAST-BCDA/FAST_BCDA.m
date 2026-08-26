function [x,fopt,ttot,iter,errVec,timeVec,res,mvp]=...
    FAST_BCDA(a,b,tau,xopt,varargin)


% FAST_BCDA version 1.0, May 10, 2014
%
% This function solves the convex problem
%
% arg min_x = 0.5*|| b - a x ||_2^2 + tau ||x||_1
%
% using the FAST_BCDA algorithm (with blocks of one/two variables), which
% is described in
% "A Fast Active Set Block Coordinate Descent Algorithm for l1-regularized least squares"
% by M. De Santis, S. Lucidi, F. Rinaldi
% SIAM Journal on Optimization, 26(1), pp. 781-809, (2016)
%
%
% -----------------------------------------------------------------------
% Copyright (2014): Marianna De Santis, Stefano Lucidi, Francesco Rinaldi
%
% FAST_BCDA is distributed under the terms
% of the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
%
%  ===== Required inputs =============
%
%  b: vector of observations
%
%  a: a is a k*n (where k is the size of b and n the size of x)
%     matrix
%
%  tau: regularization parameter (scalar)
%
%  ===== Optional inputs =============
%
%  'stopcr' = type of stopping criterion to use
%                    1 = stop when the objective function
%                        becomes equal or less than tolstop
%                    2 = stop when the norm of the difference between
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below tolstop
%                    Default = 2
%
%  'maxit' = maximum number of iterations; Default=1000
%  
%  'blvar' = number of variables per block (1 or 2); Default = 2
%
%  'epsst' =  epsilon value related to the active set estimate;
%             Default=0.0001
%
%  'tolstop' = stopping threshold; Default = 0.001
%
%  'subop' = if subop=1 do subspace optimization
%            otherwise no subspace optimization; Default= 0
%
%  'upd'   =  value related to the frequency of the residual update
%             i.e. upd=5 then active set residual update  done
%             every 5 iterations; Default=1
%
%  'memory'     = maximum number of components to be used in the nonactive
%                 set (used when fixup=0); Default = 0.05*m
%
%  'fixup'     = if fixup=0 number of components to be used in the 
%                nonactive set is min(memory,length of nonactive))
%                otherwise number of components dinamically chosen; 
%                Default = 0
%                 
%
%  'verbosity' = work silently (0) or verbosely (1); Default=0
%
% ===================================================


% ============ Outputs ==============================
%
%   x = solution of the main algorithm
%
%  fopt = final value of the objective function
%
%  ttot = elapsed CPU time
%
%  iter = number of iterations
% ====================================================


%start clock
t = cputime;
%tic

% test for number of required parametres
if (nargin-length(varargin)) ~= 4
    error('Wrong number of required parameters');
end

%dimensions of a
[m,n] = size(a);

% Set the defaults for the optional parameters
stopcr=2;
maxit=1000;
blvar=2;
epsst=0.0001;
tolstop=0.001;
subop=0;
upd=1;
memory=0.05*m;
fixup=0;
verbosity = 0;
fref = [];


% Read the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'STOPCR'
                stopcr = varargin{i+1};
            case 'MAXIT'
                maxit = varargin{i+1};
            case 'BLVAR'
                blvar = varargin{i+1}; 
            case 'EPSST'
                epsst = varargin{i+1};
            case 'TOLSTOP'
                tolstop = varargin{i+1};
            case 'SUBOP'
                subop = varargin{i+1};
            case 'UPD'
                upd = varargin{i+1};
            case 'MEMORY'
                memory = varargin{i+1};
            case 'FIXUP'
                fixup = varargin{i+1};   
            case 'VERBOSITY'
                verbosity = varargin{i+1};
            case 'FREF'
                fref = varargin{i+1};
            otherwise
                % something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end
%%%%%%%%%%%%%%



% create function handles for multiplication by A and A'
if ~isa(a, 'function_handle')
    AT = @(x) (a')*x;
end

if ~isa(a, 'function_handle')
    A = @(x) a*x;
end



%calculate norm of the columns of a
nrma=zeros(n,1);
for i=1:n
    nrma(i)=norm(a(:,i),2);
end

%initialize vectors
xold=zeros(n,1);
x=zeros(n,1);
g=zeros(n,1);
r=zeros(m,1);
sa=zeros(maxit,1);

actold=zeros(n,1);
ax=zeros(n,1);
mvp=0;


%initialize vector r =Ax-b
r=-b;


%main iteration
if (subop==1)
    sbmax=5;
else
    sbmax=0;
end

sb=0;
it=1;
dp=0.01;

while (it<=maxit)
    
    %calculate active set and nonactive set
    g=AT(r);
    mvp=mvp+1;
    res = norm(x - shringkage(x-g,tau),2);
    % Use the exact residual for objective-gap stopping. The incremental
    % residual update is useful for coordinate work, but the stopping test
    % must be based on the current iterate itself.
    if stopcr == 4
        r = a*x-b;
        mvp=mvp+1;
    end
    f = 0.5*r'*r + tau*norm(x,1);
    if (stopcr == 3 && res <= tolstop) || ...
            (stopcr == 4 && ~isempty(fref) && ...
            (f-fref)/max(1,abs(fref)) <= tolstop)
        break
    end
    
    nact= find((max(0,x)>epsst*(tau+g) ) | (max(0,-x)>epsst*(tau-g) )) ;
    [~,nacts]=sort(...
        (1.0-abs(sign(x(nact)))).*...
        max (0, max(-g(nact)-tau, g(nact)-tau))+...
        abs(sign(x(nact))).*...
        abs( g(nact) + sign(x(nact)).*tau)...
        ,'descend' );
    
    sa(it)=length(nact);
    
    act=  (max(0,x)<=epsst.*(tau+g) ) & (max(0,-x)<=epsst.*(tau-g) );
    ax=xor(act,actold);
    % Preserve the iterate used by the active-set residual update below.
    xold=x;
    x(ax)=0.0;
    
    sa(it)=length(nact);
    
    if (mod(it,upd)==0) && any(ax)
        r=r+a(:,ax)*(x(ax)-xold(ax));
    end
    
    
    %naive update of active set variables and residual r
    %act=  (max(0,x)<=epsst*(tau+g) ) & (max(0,-x)<=epsst*(tau-g) ) ;
    %x(act)=0.0;
    
    %update residual
    %if (mod(it,upd)==0)
    %     r=A(x)-b;
    %end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    if (it<=2)
        saold=0;
    else
        saold=sa(it-2);
    end
    
    if (abs(sa(it)-saold)<1) && (sa(it)<0.05*n) && (sb<sbmax)
        % subspace optimization
        display(' subspace optimization ');
        xsubold = x(nact);
        B=a(:,nact)';
        c=-B*b+sign(x(nact))*tau;
        opts = optimset('Algorithm','interior-point-convex','Display','off');
        x(nact)=quadprog(B*B',c,[],[],[],[],[],[],[],opts);
        r=r+a(:,nact)*(x(nact)-xsubold);
        sb=sb+1;
        
    else
        % block coordinate optimization 
        if (fixup==0)
            ec=min(memory,length(nact));
        else
            ec=int64(min(dp,1.0d0)*length(nact));
            dp=dp*2.0;
        end 
        
        if (blvar==1)
            % 1 coordinate optimization
            for  k=1:ec
                
                i=nact(nacts(k));
                
                xiold=x(i);
                gi=a(:,i)'*r;
                
                c=gi-nrma(i)^2*x(i);
                
                if(c< -tau)
                    x(i)=-(c+tau)/nrma(i)^2;
                    
                else if (c> tau)
                        x(i)=-(c-tau)/nrma(i)^2;
                        
                    else
                        x(i)=0.0;
                    end
                    
                end
                
                r=r+a(:,i).*(x(i)-xiold);
                
                
            end
        else
            % 2 coordinate optimization
            for  k=1:2:ec-1
                i=nact(nacts(k));
                j=nact(nacts(k+1));
                
                xiold=x(i);
                xjold=x(j);
                
                c11=nrma(i)^2;
                c22=nrma(j)^2;
                c12=a(:,i)'*a(:,j);
                c13=a(:,i)'*r -c11*xiold - c12*xjold;
                c23=a(:,j)'*r -c22*xjold - c12*xiold;
                c14=tau;
                c24=tau;
                
                kkt1=c11*c23-c12*c13;
                kkt2=c22*c13-c12*c23;
                kkt3=c11+c12;
                kkt4=c11-c12;
                kkt5=c22+c12;
                kkt6=c22-c12;
                
                delta=c11*c22-c12*c12;
                
                if ((c13>=-tau)&&(c13<=tau)&&(c23>=-tau)&&(c23<=tau))
                    x(i)=0.0;
                    x(j)=0.0;
                else if ((c23<-tau)&&(kkt2>=-tau*kkt6)&&(kkt2<=tau*kkt5))
                        x(i)=0.0;
                        x(j)=-(c23+tau)/c22;
                    else if ((c23>tau)&&(kkt2>=-tau*kkt5)&&(kkt2<=tau*kkt6))
                            x(i)=0.0;
                            x(j)=-(c23-tau)/c22;
                        else if ((c13<-tau)&&(kkt1>=-tau*kkt4)&&(kkt1<=tau*kkt3))
                                x(i)=-(c13+tau)/c11;
                                x(j)=0.0;
                            else if ((c13>tau)&&(kkt1>=-tau*kkt3)...
                                        &&(kkt1<=tau*kkt4))
                                    
                                    x(i)=-(c13-tau)/c11;
                                    x(j)=0.0;
                                else if  ((kkt1< -tau*kkt4)&&(kkt2< -tau * kkt6))
                                        
                                        x(i)=(-c22*(c13+c14)...
                                            +c12*(c23+c24))/delta;
                                        x(j)=(-c11*(c23+c24)...
                                            +c12*(c13+c14))/delta;
                                        
                                    else if  ((kkt1> tau*kkt4)...
                                                &&(kkt2> tau * kkt6))
                                            x(i)=(-c22*(c13-c14)...
                                                +c12*(c23-c24))/delta;
                                            x(j)=(-c11*(c23-c24)...
                                                +c12*(c13-c14))/delta;
                                        else if ((kkt1> tau*kkt3)...
                                                    &&(kkt2< -tau * kkt6))
                                                
                                                x(i)=(-c22*(c13+c14)...
                                                    +c12*(c23-c24))/delta;
                                                x(j)=(-c11*(c23-c24)...
                                                    +c12*(c13+c14))/delta;
                                            else if ((kkt1<-tau*kkt3)...
                                                        &&(kkt2> tau * kkt5))
                                                    
                                                    x(i)=(-c22*(c13-c14)...
                                                        +c12*(c23+c24))/delta;
                                                    x(j)=(-c11*(c23+c24)...
                                                        +c12*(c13-c14))/delta;
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                
                r=r+a(:,i).*(x(i)-xiold)+a(:,j).*(x(j)-xjold);
                
            end
        end
        
    end
    actold=act;
    
    %print info
    if (verbosity==1)
     
        f=0.5*r'*r;
        f=f+tau*norm(x,1);
        fprintf(1,'iter = %3d, nact=%3d, f= %10.3e \n'...
            ,it, length(nact), f);
   
    end
    
    
    % % compute stopping criteria and test for termination
    % switch stopcr
    %     case 1,
    %         % continue if not yeat reached target value tolstop
    %         f=0.5*r'*r;
    %         f=f+tau*norm(x,1);
    % 
    %         if (f<=tolstop)
    %             break
    %         end
    %     case 2,
    %         % stopping criterion based on relative norm of step taken
    %         if (norm(x-xold,2)/norm(x,2)<tolstop)
    %             break
    %         end
    %         xold=x;
    %     case 3,
    %         if res <= tolstop
    %             break
    %         end
    %     otherwise,
    %         error(['Unknown stopping criterion']);
    % end % end of the stopping criteria switch
    
    
    errVec(it) = ((x - xopt)'*(x - xopt))/(xopt'*xopt);
    %timeVec(it) = toc;
    timeVec(it) = cputime - t;
    it=it+1;
    
    
    
end

ttot = cputime - t;
%ttot=toc;

%print final info
r=a*x-b;
f=0.5*r'*r;
f=f+tau*norm(x,1);
fopt=f;
fprintf(1,'iter = %3d, f= %10.3e, tottime=%10.3e \n'...
    ,it, f,ttot);
iter=it;
errVec(it) = ((x - xopt)'*(x - xopt))/(xopt'*xopt);
%timeVec(it) = toc;
timeVec(it) = cputime - t;

end

function ss = shringkage(xx,mumu)
        ss = sign(xx).*max(abs(xx)-mumu,0);
end
