clc; clear all;

addpath 'IRPN'
addpath ISQA_plus
addpath 'dataset'
dataset = 'dataset/news20.binary';
[b,A] = libsvmread(dataset);
[m,n] = size(A);

x0 = zeros(n,1);
mu = 1/m;

eigsopt.issym = 1;
Rmap=@(x) A*x;
Rtmap=@(x) A'*x;
RRtmap=@(x) Rmap(Rtmap(x));
Lip = eigs(RRtmap,length(b),1,'LA',eigsopt);

tol = 1e-6;

opts = struct();
opts.verbose = 0;
opts.maxit = 500;
opts.ls = 1;
opts.bb = 1;
opts.alpha0 = 1;
opts.ftol = 1e-10;
opts.gtol = 1e-10;
[x, out] = lr_2mproj(x0, A, b, mu, opts, 2); % inexact newton CG
f_star = min(out.fvec);


% opts = struct();
% opts.verbose = 0;
% opts.maxit = 10000;
% opts.ls = 1;
% opts.bb = 1;
% opts.alpha0 = 1;
% opts.ftol = 1e-8;
% opts.gtol = tol;
% [x1, out1] = lr_proximal_grad(x0, A, b, mu, opts);
% data1 = out1.fvec-f_star;
% k1 = min(length(data1),10000);
% data1 = data1(1:k1);

opts = struct();
opts.verbose = 0;
opts.maxit = 100;
opts.ls = 1;
opts.bb = 0;
opts.alpha0 = 1;
opts.ftol = 1e-8;
opts.gtol = tol;
opts.opt_dim = out.dimension(end);
[x2, out2] = lr_2mproj(x0, A, b, mu, opts, 2);
data2 = out2.fvec-f_star;
k2 = min(length(data2),1000);
data2 = data2(1:k2);

% opts = struct();
% opts.maxit = 1000;
% opts.adp = 1;
% model.loss = 'logistic';
% model.penalty = 'ell1';
% model.regpara = mu;
% model.eps = 1e-6;
% data.A = A;
% data.b = b;
% [funv3, x3, resi3, numit3, cput3, Lipt] = alg_fista(x0, data, model, opts, ss);
% data3 = funv3-f_star;
% k3 = min(length(data3),1000);
% data3 = data3(1:k3);

opts = struct();
opts.verbose = 0;
opts.maxit = 10000;
opts.ls = 1;
opts.bb = 1;
opts.alpha0 = 1;
opts.ftol = 1e-8;
opts.gtol = tol;
opts.opt_dim = out.dimension(end);
[x4, out4] = lr_newtonacc(x0, A, b, mu, opts);
data4 = out4.fvec-f_star;
k4 = min(length(data4),1000);
data4 = data4(1:k4);

opts = struct();
opts.maxit = 100;
opts.maxitsub = 100;
opts.mu = 1e-6;
opts.eta = 0.5;
opts.rho = 0.5;    
opts.beta = 0.25;   
opts.sigma = 0.25; 
opts.ftol = 1e-10;
opts.opt_dim = out.dimension(end);
data.A = A;
data.b = b;
model.loss = 'logistic';
model.penalty = 'ell1';
model.regpara = mu;
model.eps = tol;
[funv5, x5, resi5, numit5, cput5, ~, dimension5, t_id5, itr_id5] = alg_rpn(x0, data, model, opts);
data5 = funv5-f_star;
k5 = min(length(data5),1000);
data5 = data5(1:k5);

opts = struct();
opts.maxit = 500;
opts.maxitsub = 100;
opts.mu = 1e-6;
opts.eta = 0.5;
opts.rho = 0.5;    
opts.beta = 0.25;   
opts.sigma = 1e-4; 
opts.ftol = 1e-10;
data.A = A;
data.b = b;
model.loss = 'logistic';
model.penalty = 'ell1';
model.regpara = mu;
model.eps = tol;
[x6, cput6, k6, fval6, res6] = alg_sqa(x0, data, model, opts, Lip);
data5 = funv5-f_star;
k5 = min(length(data5),1000);
data5 = data5(1:k5);

% opts = struct();
% opts.verbose = 0;
% opts.maxit = 1000;
% opts.ls = 1;
% opts.bb = 0;
% opts.alpha0 = 1;
% opts.ftol = 1e-8;
% opts.gtol = 1e-6;
% [x6, out6] = lr_2mproj_bound(x0, A, b, mu, opts);
% data6 = out6.fvec-f_star;
% k6 = min(length(data6),1000);
% data6 = data6(1:k6);

LastName = {'outer iter.';'inner iter.';'time'};
Lasttype = {'int';'int';'double'};
% PGM = [NaN; k1; out1.tt];
% FISTA = [NaN; numit3; cput3];
AltN = [NaN; out4.itr; out4.tt];
IRPN = [sum(numit5>0); sum(numit5); cput5];
TMAP = [NaN; out2.itr; out2.tt];
T = table(AltN, IRPN, TMAP,'RowNames',LastName);
input.data = T;
input.dataFormatMode = 'row';
input.dataFormat = {'%i',2,'%.2f',1};
input.tableBorders = 0;
latex = latexTable(input);

LastName = {'outer iter.';'inner iter.';'time'};
Lasttype = {'int';'int';'double'};
% PGM = [NaN; k1; out1.tt];
% FISTA = [NaN; numit3; cput3];
AltN = [NaN; out4.itr_id+1; out4.t_id];
IRPN = [itr_id5; sum(numit5(1:itr_id5)); t_id5];
TMAP = [NaN; out2.itr_id+1; out2.t_id];
T = table(AltN, IRPN, TMAP,'RowNames',LastName);
input.data = T;
input.dataFormatMode = 'row';
input.dataFormat = {'%i',2,'%.2f',1};
input.tableBorders = 0;
latex = latexTable(input);

% fig_function = figure;
% semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
% hold on
% semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',2);
% hold on
% semilogy(0:k3-1, data3, '--','Color',[0.4 0.1 0.5], 'LineWidth',2);
% hold on
% semilogy(0:k4-1, data4, '-','Color',[0.5 0.2 1], 'LineWidth',2);
% hold on
% semilogy(0:k5-1, data5, '-','Color',[0.8 0.5 0.2], 'LineWidth',2);
% hold on
% legend('proximal gradient', 'TMAP', 'FISTA', 'Newton acc', 'IRPN');
% ylabel('$f(x^k) - f^*$', 'fontsize', 14, 'interpreter', 'latex');
% xlabel('iteration step');

% 
% k1 = min(length(out1.dimension),100);
% k2 = min(length(out2.dimension),100);
% k3 = min(length(out3.dimension),100);
% k4 = min(length(out4.dimension),100);
% k5 = min(length(dimension5),100);
% 
% 
% fig_dimension = figure;
% semilogy(0:k1-1, out1.dimension(1:k1), '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
% hold on
% semilogy(0:k2-1, out2.dimension(1:k2), '-.','Color',[0.99 0.1 0.2], 'LineWidth',2);
% hold on
% semilogy(0:k3-1, out3.dimension(1:k3), '--','Color',[0.4 0.1 0.5], 'LineWidth',2);
% hold on
% semilogy(0:k4-1, out4.dimension(1:k4), '-','Color',[0.5 0.2 1], 'LineWidth',2);
% hold on
% semilogy(0:k5-1, dimension5(1:k5), '-','Color',[0.8 0.5 0.2], 'LineWidth',2);
% hold on
% legend('proximal gradient', 'Two-metric adaptive projection', 'FISTA', 'Newton acc', 'IRPN');
% ylabel('dimension of manifold at $x^k$', 'fontsize', 14, 'interpreter', 'latex');
% xlabel('iteration step');