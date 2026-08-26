function[x,tottime,errdecx,lsq, iter,fopt,errVec,timeVec] =...
    Call_FAST_2CDA(A,b,xsol,tau,fsparsa,varargin)
       

[x,fopt,tottime,iter,errVec,timeVec]=...
    FAST_BCDA(A,b,tau,xsol, 'tolstop', 0.01, 'blvar', 2);


%error with respect to the real solution
den = xsol'*xsol;
r = xsol - x;
num= r'*r;
errdecx=num/den;

fprintf(1,'Errorx %10.3e\n',errdecx);


%ls error
resid2 = A * x - b;
lsq = 0.5*resid2'*resid2;
fprintf(1,'0.5*||A x - y ||^2 = %10.3e\n',0.5*resid2'*resid2);

fprintf(1,'Number of non-zero components of x = %d\n',...
          sum((abs(x)>=0.0001)));
            
     
fprintf(1,'CPU time so far = %10.3e\n', tottime);
%figure;
%plot(xsol), hold on; plot(x, 'r'), hold on;
%hold off;

end 

