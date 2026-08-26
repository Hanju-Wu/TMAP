% clear;
% 
% 
% rho= 0.01;
% n = 2^14;
% % random +/- 1 signal
% x = zeros(n,1);
% q = randperm(n);
% 
% 
% disp('Creating measurment matrix and vector b...');
% 
% % number of rows
% m=n/4;
% % generation of matrix A 
% Bm=sprand(m,n,0.1);
% A=full(Bm);
% Bm=[];
% for j = 1:n
%     if(sum(abs(A(:,j)))~=0)
%         A(:,j) = A(:,j)/norm(A(:,j));
%     end
% end
% % Built solution
% % number of spikes
% T=int64(rho*m);% %0.05 e 0.01
% x(q(1:T)) = sign(randn(T,1));
% % noisy observations
% sigma = 0.001;
% e = sigma*randn(m,1);
% b = A*x + e;
% tau = 0.1*norm(A'*b,'inf');

rho = 0.01;

% signal size
N       = 2^14;
% number of measurements
M       = floor(N/4);
% number of nonzeros
K       = floor(M*rho);

T=int64(rho*M);
tol = 1e-10;
% parpool(2);
maxit = 100;
times = zeros(maxit,4);

for i=1:maxit
A = 1/sqrt(2*N)*randn(M,N);

e = 1e-2*randn(M,1); %noise
x = sign(sprandn(N,1,K/N));
b = A*x + e;

tau = 0.1*norm(A'*b,'inf');


disp('Done.');

verbosity=0;
delta=0.8;



% display('***********');
% display(' FAST-2CDA ');
% display('***********');
% 
% 
% [xour2,tottime2,err2,errlsq2, iter2, fsparsa,...
%     errVec2, timeVec2] = ...
%     Call_FAST_2CDA(A,b, x, tau,...
%     fsparsa,int64(0.8*delta*T));
% 
% 
% 
% display('***********');
% display(' FAST-1CDA ');
% display('***********');
% 
% 
% [xour1,tottime1,err1,errlsq1, iter1, f1,...
%     errVec1, timeVec1] =  ...
%     Call_FAST_1CDA(A,b, x, tau,...
%     fsparsa,int64(0.8*delta*T));
% 
% 
% 
% 
% display('***************');
% display(' FAST-1CDA_ehn ');
% display('***************');
% 
% [xour1acc,tottime1acc,err1acc,errlsq1acc,...
%     iter1acc, f1acc, errVec1acc, timeVec1acc] = ...
%     Call_FAST_1CDA_ehn(A,b, x, tau,...
%     fsparsa,int64(0.8*delta*T));


display('***************');
display(' FAST-2CDA_ehn ');
display('***************');
tic;

[xour2acc,tottime2acc,err2acc,errlsq2acc,...
    iter2acc, f2acc, errVec2acc, timeVec2acc,res2acc] =  ...
    Call_FAST_2CDA_ehn(A,b, x, tau,...
    tol,int64(0.8*delta*T));

t2acc = toc;


x0 = zeros(N,1);
mu = tau;

opts = struct();
opts.maxit = 1000;
opts.c = 1e-2;
opts.M = 5;
opts.ssu = 1e+30;
opts.ssl = 1e-30;
opts.beta = 0.2;   
opts.tol = tol;
[funv1, x1, resi1, numit1, cput1] = alg_nmt(A, b, x0, mu, opts);



opts = struct();
opts.maxit = 1000;
opts.tol = tol;
opts.x0 = x0;
opts.crit = 1;
opts.cont = 1;
[x2,out2]     = tmap(A,b,N,mu,opts);

opts.cont = 0;
[x3,out3]     = tmap(A,b,N,mu,opts);

times(i,:) = [t2acc cput1 out2.time out3.time];

end

figure;
boxplot(times, 'Labels', {'FAST-2CDA-E','SpaRSA','TMAP-AC','TMAP'});
ylabel('CPU time (s)');
grid on;