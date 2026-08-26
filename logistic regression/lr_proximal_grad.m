function [x, out] = lr_proximal_grad(x0, A, b, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 0; end

out = struct();
[m,~] = size(A);
k = 0;
x = x0;
t = opts.alpha0;
fp = inf;

tt = tic;

Ax = A*x;
expba = exp(- b.*Ax);
f = sum(log(1 + expba))/m + mu*norm(x,1);
g = A'*(b./(1+expba) - b)/m;
tmpf = f;
nrmG = norm(x - prox(x - g,mu),2);
out.fvec = f;
out.dimension = sum(abs(x)>0);

%%%
% 线搜索参数。
Cval = tmpf; Q = 1; gamma = 0.85; rhols = 1e-6;
%% 迭代主循环
% 当达到最大迭代次数，或梯度或函数值的变化大于阈值时，退出迭代。
while k < opts.maxit && nrmG > opts.gtol %&& abs(f - fp) > opts.ftol
    %%%
    % 记录上一步的迭代信息。
    gp = g;
    fp = f;
    xp = x;
    
    %%%
    % 一步近似点梯度法。令 $\phi(x)=\frac{1}{2}\|Ax-b\|_2^2$, $h(x)=\mu\|x\|_1$，
    % 近似点梯度法的迭代格式为
    % $x^{k+1}=\mathrm{prox}_{t_{k}h}(x^k-t_{k}A^\top(Ax^k-b))$，
    % 近邻算子 |prox| 的计算见辅助函数。
    x = prox(xp - t * g, t * mu);
    %%%
    % 事实上，近似点梯度法的迭代格式根据定义可以写作
    %
    % $$ \begin{array}{ll} x^{k+1}&\hspace{-0.5em}=\displaystyle\arg\min_u
    % \left( \|u\|_1+\frac{1}{2 t_k}\|u-x^k+ t_k\nabla \phi(x^k)\|_2^2 \right)  \\
    % &\hspace{-0.5em}=\displaystyle\arg\min_u \left( \|u\|_1+\phi(x^k)
    % +\nabla \phi(x^k)^\top (u-x^k)+\frac{1}{2 t_k}\|u-x^k\|^2_2 \right).
    % \end{array} $$
    %
    %%%
    % 检验是否满足非精确线搜索条件。
    % 令 $f(x) = \phi(x) + h(x)$，针对 $f(x)$ 考虑线搜索准则，即为 $f(x^{k+1}(t))\le C_k - \frac{1}{2}\rho t
    % \| x^{k+1}(t)-x^k\|^2$，其中 $x^{k+1}(t) = \mathrm{prox}_{t h}(x^k - t \nabla \phi(x^k))$。 
    %
    % |nls| 记录线搜索循环的迭代次数，
    % 直到满足条件或进行
    % 5 次步长衰减后退出线搜索循环，得到更新的 $x^{k+1}$。 $C_k$ 为 (Zhang &
    % Hager) 线搜索准则中的量。
    %
    % 如果不满足线搜索条件，对当前步长进行衰减，当前线搜索次数加一。
    if opts.ls
        nls = 0;
        while 1
            Ax = A*x;
            expba = exp(- b.*Ax);
            tmpf = sum(log(1 + expba))/m + mu*norm(x,1);

            if tmpf <= Cval - rhols*0.5*t*norm(x-xp,2)^2 || nls == 5
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(xp - t * g, t * mu);
        end
        
        f = tmpf;
        %%%
        % 当 opts.ls=0 时，不进行线搜索。
    else
        Ax = A*x;
        expba = exp(- b.*Ax);
        f = 0.5 * norm(A*x - b, 2)^2 + mu*norm(x,1);
    end
    g = A'*(b./(1+expba) - b)/m;
    out.dimension = [out.dimension, sum(abs(x)>0)];

    nrmG = norm(x - prox(x - g,mu),2);
    
    
    %%%
    % 如果 |opts.bb=1| 且 |opts.ls=1| 则计算 BB 步长作为下一步迭代的初始步长。令
    % $s^k=x^{k+1}-x^k$, $y^k=g^{k+1}-g^k$，
    % 这里在偶数与奇数步分别对应 $\displaystyle\frac{(s^k)^\top s^k}{(s^k)^\top y^k}$
    % 和 $\displaystyle\frac{(s^k)^\top y^k}{(y^k)^\top y^k}$ 两个 BB 步长。
    if opts.bb && opts.ls
        dx = x - xp;
        dg = g - gp;
        dxg = abs(dx'*dg);
        
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,2)^2/dxg;
            else
                t = dxg/norm(dg,2)^2;
            end
        end
        
        %%%
        % 将更新得到的 BB 步长限制在阈值 [t_0,10^{12}] 内。
        t = min(max(t,opts.alpha0),1e12);
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmpf)/Q;
        
        %%%
        % 如果不使用 BB 步长，则使用设定的初始步长开始下一次迭代。
    else
        t = opts.alpha0;
    end
    
    %%%
    % 迭代步数加一，记录当前函数值，输出信息。
    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end
    
    %%%
    % 特别地，除了每次迭代开始处的收敛条件外，如果连续 8 步的函数值最小值比 8 步之前的函数值超过阈值，
    % 则停止内层循环。
    if k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break;
    end
end

out.fvec = out.fvec(1:k);
out.dimension = out.dimension(1:k);
out.fval = f;
out.itr = k;
out.tt = toc(tt);
out.nrmG = nrmG;
end
%% 辅助函数
% 函数 $h(x)=\mu\|x\|_1$ 对应的邻近算子 $\mathrm{sign}(x)\max\{|x|-\mu,0\}$。
function y = prox(x, mu)
y = max(abs(x) - mu, 0);
y = sign(x) .* y;
end
%% 参考页面
% 该函数由连续化策略调用，关于连续化策略参见
% <..\LASSO_con\LASSO_con.html LASSO问题连续化策略>。
%
% 在页面 <demo_proxg.html 实例：近似点梯度法和 Nesterov 加速算法求解 LASSO 问题>
% 我们展示该算法的应用。另外，基于该算法的加速算法参考
% <LASSO_Nesterov_inn.html LASSO问题的 FISTA
% 算法>、 <LASSO_Nesterov2nd_inn.html LASSO问题的第二类 Nesterov 加速算法>。
%
% 此页面的源代码请见：
% <../download_code/lasso_proxg/LASSO_proximal_grad_inn.m
% LASSO_proximal_grad_inn.m>。
%% 版权声明
% 此页面为《最优化：建模、算法与理论》、《最优化计算方法》配套代码。
% 代码作者：文再文、刘浩洋、户将，代码整理与页面制作：杨昊桐。
%
% 著作权所有 (C) 2020 文再文、刘浩洋、户将