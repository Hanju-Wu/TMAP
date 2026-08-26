function [w,out] = L1General2_TMP(funObj,w,lambda,options)
%L1GENERAL2_TMP Two-metric projection method with performance tracing.
% The output fields follow L1General2_PSSgb.  fvec is objectiveScale times
% the native TMP objective and fvec_native contains the internal values.

%% Process Options
if nargin < 4
    options = [];
end
tt = tic;

[verbose,optTol,progTol,maxIter,suffDec,corrections,lossType,mvpPerEval, ...
    objectiveScale,stopCriterion,fReference] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'corrections',100,'lossType','auto', ...
    'mvpPerEval',[],'objectiveScale',[],'stopCriterion','proxResidual', ...
    'fReference',[]);

[lossType,mvpPerEval] = resolveMvpCost( ...
    funObj,lossType,mvpPerEval,nargout > 1);
objectiveScale = resolveObjectiveScale(lossType,objectiveScale);
stopCriterion = resolveStopCriterion(stopCriterion);
if strcmp(stopCriterion,'relativeFunctionGap') && ...
        (isempty(fReference) || ~isscalar(fReference) || ~isfinite(fReference))
    error('options.fReference must be a finite scalar for relativeFunctionGap stopping.');
end

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
w = [w.*(w>0);-w.*(w<0)];
[f,g] = nonNegGrad(funObj,w,p,lambda);
funEvals = 1;
acceptedIter = 0;
out = initializeOutput( ...
    f,funEvals,lossType,mvpPerEval,objectiveScale,w,p,toc(tt));

% Compute working set and check optimality
W = (w~=0) | (g < 0);
optCond = max(abs(g(W)));
out.optCond = optCond;
initialGap = (objectiveScale*f-fReference)/max(1,abs(fReference));
if (strcmp(stopCriterion,'proxResidual') && optCond < optTol) || ...
        (strcmp(stopCriterion,'relativeFunctionGap') && initialGap <= optTol)
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    w = w(1:p)-w(p+1:end);
    return;
end

%% Main loop
for i = 1:maxIter
    out.outerIter = i;
    
    % Compute direction
    d = zeros(2*p,1);
    if i == 1
        d(W) = -g(W);
        Y = zeros(2*p,0);
        S = zeros(2*p,0);
        sigma = 1;
        t = min(1,1/sum(abs(g)));
    else
        y = g-g_old;
        s = w-w_old;
        
        correctionsStored = size(Y,2);
        if correctionsStored < corrections
            Y(:,correctionsStored+1) = y;
            S(:,correctionsStored+1) = s;
        else
            Y = [Y(:,2:corrections) y];
            S = [S(:,2:corrections) s];
        end
        
        ys = y'*s;
        if ys > 1e-10
            sigma = ys/(y'*y);
        end
        
        curvSat = sum(Y(W,:).*S(W,:)) > 1e-10;
        
        d(W) = lbfgsC(-g(W),S(W,curvSat),Y(W,curvSat),sigma);
        t = 1;
    end
    f_old = f;
    g_old = g;
    w_old = w;
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    % Compute projected point
    w_new = nonNegProject(w+t*d);
    [f_new,g_new] = nonNegGrad(funObj,w_new,p,lambda);
    funEvals = funEvals+1;
    
    % Line search along projection arc
    while f_new > f + suffDec*g'*(w_new-w) || ~isLegal(f_new)
        t_old = t;
        
        % Backtracking
        if verbose
            fprintf('Backtracking...\n');
        end
        if ~isLegal(f_new)
            if verbose
                fprintf('Halving Step Size\n');
            end
            t = .5*t;
        else
            t = polyinterp([0 f gtd; t f_new g_new'*d]);
        end
        
        % Adjust if interpolated value near boundary
        if t < t_old*1e-3
            if verbose == 3
                fprintf('Interpolated value too small, Adjusting\n');
            end
            t = t_old*1e-3;
        elseif t > t_old*0.6
            if verbose == 3
                fprintf('Interpolated value too large, Adjusting\n');
            end
            t = t_old*0.6;
        end
        
        % Check whether step has become too small
        if max(abs(t*d)) < progTol
            if verbose
                fprintf('Step too small in line search\n');
            end
            t = 0;
            w_new = w;
            f_new = f;
            g_new = g;
            break;
        end
        
        % Compute projected point
        w_new = nonNegProject(w+t*d);
        [f_new,g_new] = nonNegGrad(funObj,w_new,p,lambda);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    stepAccepted = t > 0;
    if stepAccepted
        acceptedIter = acceptedIter + 1;
    end
    out = appendOutput( ...
        out,f,funEvals,mvpPerEval,objectiveScale,w,p,acceptedIter, ...
        stepAccepted,toc(tt));
    
    % Compute new working set
    W = (w~=0) | (g < 0);
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(g(W))),nnz(w(1:p)-w(p+1:end)));
    end
    
    % Check Optimality
    optCond = max(abs(g(W)));
    out.optCond = optCond;
    functionGap = (objectiveScale*f-fReference)/max(1,abs(fReference));
    if (strcmp(stopCriterion,'proxResidual') && optCond < optTol) || ...
            (strcmp(stopCriterion,'relativeFunctionGap') && functionGap <= optTol)
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end
    
    % Check for lack of progress
    if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
        if verbose
        fprintf('Progress in parameters or objective below progTol\n');
        end
        break;
    end
    
    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        break;
    end
end
w = w(1:p)-w(p+1:end);

out.funEvals = funEvals;
out.iter = acceptedIter;
out.num_Ax_total = mvpPerEval(1)*funEvals;
out.num_ATy_total = mvpPerEval(2)*funEvals;
out.mvp_total = sum(mvpPerEval)*funEvals;
out.nnz = nnz(w);
out.runtime = toc(tt);

end

%% Psuedo-gradient calculation
function [f,g,H] = nonNegGrad(funObj,w,p,lambda)

[f,g] = funObj(w(1:p)-w(p+1:end));

f = f + sum(lambda.*w(1:p)) + sum(lambda.*w(p+1:end));

g = [g;-g] + [lambda.*ones(p,1);lambda.*ones(p,1)];
end

function [w] = nonNegProject(w)
w(w < 0) = 0;
end

function [lossType,mvpPerEval] = resolveMvpCost( ...
        funObj,lossType,mvpPerEval,requireCount)
if isstring(lossType); lossType = char(lossType); end
if isempty(lossType); lossType = 'auto'; end
lossType = lower(strtrim(lossType));

if strcmp(lossType,'auto')
    objectiveName = lower(func2str(funObj));
    if ~isempty(strfind(objectiveName,'logisticloss')) %#ok<STREMP>
        lossType = 'logistic';
    elseif ~isempty(strfind(objectiveName,'squarederror')) %#ok<STREMP>
        lossType = 'lasso';
    else
        lossType = 'unknown';
    end
end

if isempty(mvpPerEval)
    if any(strcmp(lossType,{'logistic','logistic-regression', ...
            'logistic_regression','lasso','least-squares', ...
            'least_squares','squared-error','squared_error', ...
            'squarederror'}))
        mvpPerEval = [1,1];
    elseif requireCount
        error(['Cannot infer matrix-product cost for this objective. ', ...
            'Set options.lossType to ''logistic'' or ''lasso'', or ', ...
            'set options.mvpPerEval to [numForward,numTranspose].']);
    else
        mvpPerEval = [NaN,NaN];
    end
elseif ~isnumeric(mvpPerEval) || numel(mvpPerEval) ~= 2 || ...
        any(~isfinite(mvpPerEval)) || any(mvpPerEval < 0)
    error('options.mvpPerEval must be [numForward,numTranspose].');
else
    mvpPerEval = reshape(mvpPerEval,1,2);
end
end

function objectiveScale = resolveObjectiveScale(lossType,objectiveScale)
if isempty(objectiveScale)
    if any(strcmp(lossType, ...
            {'lasso','least-squares','least_squares','squared-error', ...
            'squared_error','squarederror'}))
        objectiveScale = 0.5;
    else
        objectiveScale = 1;
    end
elseif ~isnumeric(objectiveScale) || ~isscalar(objectiveScale) || ...
        ~isfinite(objectiveScale) || objectiveScale <= 0
    error('options.objectiveScale must be a positive finite scalar.');
end
end

function criterion = resolveStopCriterion(criterion)
if isstring(criterion); criterion = char(criterion); end
if ~ischar(criterion)
    error('options.stopCriterion must be ''proxResidual'' or ''relativeFunctionGap''.');
end
key = lower(strtrim(criterion));
if any(strcmp(key,{'proxresidual','prox-residual','prox_residual'}))
    criterion = 'proxResidual';
elseif any(strcmp(key,{'relativefunctiongap','relative-function-gap', ...
        'relative_function_gap','functiongap','function-gap'}))
    criterion = 'relativeFunctionGap';
else
    error('options.stopCriterion must be ''proxResidual'' or ''relativeFunctionGap''.');
end
end

function out = initializeOutput( ...
        f,funEvals,lossType,mvpPerEval,objectiveScale,w,p,elapsedTime)
out = struct();
out.lossType = lossType;
out.mvpPerEval = mvpPerEval;
out.objectiveScale = objectiveScale;
out.fvec_native = f;
out.fvec = objectiveScale*f;
out.funEvalsTrace = funEvals;
out.num_Ax = mvpPerEval(1)*funEvals;
out.num_ATy = mvpPerEval(2)*funEvals;
out.mvp = sum(mvpPerEval)*funEvals;
out.timevec = elapsedTime;
out.funEvals = funEvals;
out.iter = 0;
out.outerIter = 0;
out.isAcceptedPoint = true;
out.num_Ax_total = out.num_Ax(end);
out.num_ATy_total = out.num_ATy(end);
out.mvp_total = out.mvp(end);
out.optCond = NaN;
out.nnzvec = nnz(w(1:p)-w(p+1:end));
out.nnz = nnz(w(1:p)-w(p+1:end));
out.runtime = elapsedTime;
end

function out = appendOutput( ...
        out,f,funEvals,mvpPerEval,objectiveScale,w,p,iter,stepAccepted, ...
        elapsedTime)
out.fvec_native(end+1,1) = f;
out.fvec(end+1,1) = objectiveScale*f;
out.funEvalsTrace(end+1,1) = funEvals;
out.num_Ax(end+1,1) = mvpPerEval(1)*funEvals;
out.num_ATy(end+1,1) = mvpPerEval(2)*funEvals;
out.mvp(end+1,1) = sum(mvpPerEval)*funEvals;
out.timevec(end+1,1) = elapsedTime;
out.isAcceptedPoint(end+1,1) = stepAccepted;
out.funEvals = funEvals;
out.iter = iter;
out.num_Ax_total = out.num_Ax(end);
out.num_ATy_total = out.num_ATy(end);
out.mvp_total = out.mvp(end);
out.nnzvec(end+1,1) = nnz(w(1:p)-w(p+1:end));
out.nnz = nnz(w(1:p)-w(p+1:end));
out.runtime = elapsedTime;
end
