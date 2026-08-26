function [w,out] = L1General2_PSSgb(funObj,w,lambda,options)
%L1GENERAL2_PSSGB Projected-scaled subgradient method with CG or L-BFGS.
% For LogisticLoss and SquaredError, each [f,g] evaluation performs one
% data-matrix product and one transpose data-matrix product.  Use
% options.lossType to override automatic detection, or options.mvpPerEval
% to supply [numForward,numTranspose] for a custom objective.  The fvec
% trace is objectiveScale times the native objective; fvec_native retains
% the values used internally by PSSgb.
%
% The default inner solver is the original L-BFGS implementation. To use
% CG, set options.inner_solver = 'cg' (innerSolver is also accepted).
% SquaredError and LogisticLoss operators are built automatically from the
% data captured by funObj. options.cgOperator remains available as an
% override for custom smooth objectives.

%% Process Options
if nargin < 4
    options = struct();
elseif isempty(options)
    options = struct();
end
options = normalizeOptionAliases(options);
tt = tic;

[verbose,optTol,progTol,maxIter,suffDec,corrections,Dtype,quadraticInit, ...
    lossType,mvpPerEval,objectiveScale,innerSolver,cgOperator,cgTol, ...
    cgMaxIter,cgAdapt,cgRegularization,cgRegularizationDecay, ...
    cgRegularizationMin, ...
    cgPreconditioner,mvpPerHvp,stopCriterion,fReference] = ...
    myProcessOptions(options,'verbose',1,'optTol',1e-5,'progTol',1e-9,...
    'maxIter',500,'suffDec',1e-4,'corrections',100,'Dtype',1, ...
    'quadraticInit',0,'lossType','auto','mvpPerEval',[], ...
    'objectiveScale',[],'innerSolver','lbfgs','cgOperator',[], ...
    'cgTol',1e-1,'cgMaxIter',5,'cgAdapt',1, ...
    'cgRegularization',1e-4,'cgRegularizationDecay',0.2, ...
    'cgRegularizationMin',1e-12, ...
    'cgPreconditioner',[],'mvpPerHvp',[], ...
    'stopCriterion','pseudoGradient','fReference',[]);

[lossType,mvpPerEval] = resolveMvpCost( ...
    funObj,lossType,mvpPerEval,nargout > 1);
objectiveScale = resolveObjectiveScale(lossType,objectiveScale);
[innerSolver,useCG] = resolveInnerSolver(innerSolver);
[cgOperator,cgTol,cgMaxIter,cgAdapt,cgRegularization, ...
    cgRegularizationDecay,cgRegularizationMin,cgPreconditioner, ...
    mvpPerHvp] = ...
    validateCgOptions(cgOperator,cgTol,cgMaxIter,cgAdapt, ...
    cgRegularization,cgRegularizationDecay,cgRegularizationMin, ...
    cgPreconditioner,mvpPerHvp,lossType);
cgOperator = resolveCgOperator(cgOperator,funObj,lossType,length(w),useCG);
stopCriterion = resolveStopCriterion(stopCriterion);
if strcmp(stopCriterion,'relativeFunctionGap') && ...
        (isempty(fReference) || ~isscalar(fReference) || ~isfinite(fReference))
    error('options.fReference must be a finite scalar for relativeFunctionGap stopping.');
end
currentCgTol = cgTol;
currentCgMaxIter = cgMaxIter;
currentCgRegularization = cgRegularization;
nextCgTightening = 1e-3;
nextRegularizationReduction = 1e-1;

if verbose
    fprintf('%6s %6s %12s %12s %12s %6s\n','Iter','fEvals','stepLen','fVal','optCond','nnz');
end

%% Evaluate Initial Point
p = length(w);
[f,g,smoothGrad] = pseudoGrad(funObj,w,lambda);
proxResidual = computeProxResidual( ...
    w,smoothGrad,lambda,objectiveScale);
res = norm(proxResidual);
funEvals = 1;
acceptedIter = 0;
out = initializeOutput( ...
    f,funEvals,lossType,mvpPerEval,objectiveScale,w,toc(tt), ...
    innerSolver,mvpPerHvp,res,stopCriterion);
numHvp = 0;

% Check optimality
optCond = max(abs(g));
stopValue = computeStopValue(stopCriterion,optCond,res,objectiveScale*f,fReference);
out.optCond = optCond;
if stopValue < optTol
    if verbose
        fprintf('First-order optimality satisfied at initial point\n');
    end
    return;
end

%% Main loop
for i = 1:maxIter
    out.outerIter = i;
    cgRanThisIteration = false;
    
    % Compute active set
    W = lambda ==0 | w ~=0;
    A = W==0;
    
    % Compute direction
    d = zeros(p,1);
    if i == 1
        d = -g;
        if ~useCG
            Y = zeros(p,0);
            S = zeros(p,0);
        end
        sigma = 1;
        t = min(1,1/sum(abs(g)));
    else
        if useCG
            sig = currentCgRegularization;
            if any(W)
                hessOp = cgOperator(w,W);
                if ~isa(hessOp,'function_handle')
                    error('options.cgOperator must return a function handle.');
                end
                precf = makeCgPreconditioner(cgPreconditioner,w,W,sig);
                [d(W),cgIts,cgInfo] = mypcg_pssgb( ...
                    hessOp,-g(W),currentCgTol,currentCgMaxIter,precf,sig);
                cgRanThisIteration = true;
                numHvp = numHvp + cgIts;
                out.cg(i,1) = cgIts;
                out.cgInfo(i,1) = cgInfo;
                out.cgTol(i,1) = currentCgTol;
                out.cgMaxIter(i,1) = currentCgMaxIter;
                out.cgRegularization(i,1) = sig;
                if any(~isfinite(d(W))) || g(W)'*d(W) >= 0
                    d(W) = -g(W);
                    out.cgInfo(i,1) = 5;
                end
            end
            sigma = 1;
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

            % Keep only pairs with positive curvature on this working set.
            curvSat = sum(Y(W,:).*S(W,:)) > 1e-10;
            d(W) = lbfgsC(-g(W),S(W,curvSat),Y(W,curvSat),sigma);
        end
        if Dtype == 0
            D = min(1,1/sum(abs(g)));
        else
            D = sigma;
        end
        d(A) = -D.*g(A);
        t = 1;
    end
    g_old = g;
    w_old = w;
    
    % Compute desired orthant
    xi = sign(w);
    xi(w==0) = sign(-g(w==0));
    
    % Compute directional derivative, check that we can make progress
    gtd = g'*d;
    if gtd > -progTol
        if verbose
            fprintf('Directional derivative below progTol\n');
        end
        break;
    end
    
    if quadraticInit
        if i > 1
            t = min(1,2*(f-f_prev)/gtd);
        end
        f_prev = f;
    end
    
    % Compute projected point
    w_new = orthantProject(w+t*d,xi);
    [f_new,g_new,smoothGrad_new] = pseudoGrad(funObj,w_new,lambda);
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
            smoothGrad_new = smoothGrad;
            break;
        end
        
        % Compute projected point
        w_new = orthantProject(w+t*d,xi);
        [f_new,g_new,smoothGrad_new] = pseudoGrad(funObj,w_new,lambda);
        funEvals = funEvals+1;
    end
    
    % Take step
    w = w_new;
    f = f_new;
    g = g_new;
    smoothGrad = smoothGrad_new;
    proxResidual = computeProxResidual( ...
        w,smoothGrad,lambda,objectiveScale);
    res = norm(proxResidual);
    stepAccepted = t > 0;
    if stepAccepted
        acceptedIter = acceptedIter + 1;
    end
    out = appendOutput( ...
        out,f,funEvals,mvpPerEval,objectiveScale,w,acceptedIter, ...
        stepAccepted,toc(tt),numHvp,mvpPerHvp,res);
    
    % Output Log
    if verbose
        fprintf('%6d %6d %8.5e %8.5e %8.5e %6d\n',i,funEvals,t,f,max(abs(g)),nnz(w));
    end
    
    % Check Optimality
    optCond = max(abs(g));
    stopValue = computeStopValue(stopCriterion,optCond,res,objectiveScale*f,fReference);
    out.optCond = optCond;
    if stopValue < optTol
        if verbose
            fprintf('First-order optimality below optTol\n');
        end
        break;
    end

    if useCG && cgRanThisIteration
        while res < nextRegularizationReduction && ...
                currentCgRegularization > cgRegularizationMin
            currentCgRegularization = max(cgRegularizationMin, ...
                cgRegularizationDecay*currentCgRegularization);
            nextRegularizationReduction = ...
                0.1*nextRegularizationReduction;
        end
        if cgAdapt
            if res < nextCgTightening
                currentCgTol = max(0.1*currentCgTol,1e-7);
                nextCgTightening = 0.1*nextCgTightening;
            end
            if res < 1e-3
                currentCgMaxIter = min(25,p);
            elseif res < 1e-2
                currentCgMaxIter = min(20,p);
            elseif res < 1e-1
                currentCgMaxIter = min(15,p);
            else
                currentCgMaxIter = min(10,p);
            end
        end
    end
    
    % % Check for lack of progress
    % if max(abs(t*d)) < progTol || abs(f-f_old) < progTol
    %     if verbose
    %     fprintf('Progress in parameters or objective below progTol\n');
    %     end
    %     break;
    % end
    
    % Check for iteration limit
    if funEvals >= maxIter
        if verbose
            fprintf('Function evaluations reached maxIter\n');
        end
        break;
    end
end

out.funEvals = funEvals;
out.iter = acceptedIter;
out.numHvp = numHvp;
out.nr_CG = numHvp;
out.num_Ax_total = mvpPerEval(1)*funEvals + mvpPerHvp(1)*numHvp;
out.num_ATy_total = mvpPerEval(2)*funEvals + mvpPerHvp(2)*numHvp;
out.mvp_total = out.num_Ax_total + out.num_ATy_total;
out.nnz = nnz(w);
out.runtime = toc(tt);

end

function options = normalizeOptionAliases(options)
if ~isstruct(options)
    error('options must be a structure.');
end
aliases = { ...
    'inner_solver','innerSolver'; ...
    'CG_tol','cgTol'; ...
    'CG_maxit','cgMaxIter'; ...
    'CG_adapt','cgAdapt'; ...
    'stop_criterion','stopCriterion'};
for j = 1:size(aliases,1)
    source = aliases{j,1};
    target = aliases{j,2};
    if isfield(options,source) && ~isfield(options,target)
        options.(target) = options.(source);
    end
end
end

function stopCriterion = resolveStopCriterion(stopCriterion)
if isstring(stopCriterion); stopCriterion = char(stopCriterion); end
if ~ischar(stopCriterion)
        error(['options.stopCriterion must be ''pseudoGradient'', ', ...
            '''proxResidual'', or ''relativeFunctionGap''.']);
end
value = lower(strtrim(stopCriterion));
if any(strcmp(value,{'pseudogradient','pseudo-gradient','pseudo_gradient'}))
    stopCriterion = 'pseudoGradient';
elseif any(strcmp(value,{'proxresidual','prox-residual','prox_residual'}))
    stopCriterion = 'proxResidual';
elseif any(strcmp(value,{'relativefunctiongap','relative-function-gap', ...
        'relative_function_gap','functiongap','function-gap'}))
    stopCriterion = 'relativeFunctionGap';
else
    error(['options.stopCriterion must be ''pseudoGradient'', ', ...
        '''proxResidual'', or ''relativeFunctionGap''.']);
end
end

function stopValue = computeStopValue(stopCriterion,optCond,res,fValue,fReference)
if strcmp(stopCriterion,'proxResidual')
    stopValue = res;
elseif strcmp(stopCriterion,'relativeFunctionGap')
    stopValue = (fValue-fReference)/max(1,abs(fReference));
else
    stopValue = optCond;
end
end

function [innerSolver,useCG] = resolveInnerSolver(innerSolver)
if isstring(innerSolver); innerSolver = char(innerSolver); end
if ~ischar(innerSolver)
    error('options.innerSolver must be ''lbfgs'' or ''cg''.');
end
innerSolver = lower(strtrim(innerSolver));
if any(strcmp(innerSolver,{'lbfgs','l-bfgs'}))
    innerSolver = 'lbfgs';
    useCG = false;
elseif any(strcmp(innerSolver,{'cg','pcg'}))
    innerSolver = 'cg';
    useCG = true;
else
    error('options.innerSolver must be ''lbfgs'' or ''cg''.');
end
end

function [cgOperator,cgTol,cgMaxIter,cgAdapt,cgRegularization, ...
    cgRegularizationDecay,cgRegularizationMin,cgPreconditioner, ...
    mvpPerHvp] = ...
    validateCgOptions(cgOperator,cgTol,cgMaxIter,cgAdapt, ...
    cgRegularization,cgRegularizationDecay,cgRegularizationMin, ...
    cgPreconditioner,mvpPerHvp,lossType)
if ~isempty(cgOperator) && ~isa(cgOperator,'function_handle')
    error('options.cgOperator must be empty or a function handle.');
end

if ~isscalar(cgTol) || ~isfinite(cgTol) || cgTol <= 0 || cgTol >= 1
    error('options.cgTol must be in (0,1).');
end
if ~isscalar(cgMaxIter) || cgMaxIter < 1 || cgMaxIter ~= floor(cgMaxIter)
    error('options.cgMaxIter must be a positive integer.');
end
if ~isscalar(cgAdapt)
    error('options.cgAdapt must be scalar.');
end
cgAdapt = logical(cgAdapt);
if ~isscalar(cgRegularization) || ~isfinite(cgRegularization) || ...
        cgRegularization <= 0
    error('options.cgRegularization must be positive and finite.');
end
if ~isscalar(cgRegularizationDecay) || ...
        ~isfinite(cgRegularizationDecay) || cgRegularizationDecay <= 0 || ...
        cgRegularizationDecay >= 1
    error('options.cgRegularizationDecay must be in (0,1).');
end
if ~isscalar(cgRegularizationMin) || ~isfinite(cgRegularizationMin) || ...
        cgRegularizationMin <= 0 || cgRegularizationMin > cgRegularization
    error(['options.cgRegularizationMin must be positive and no larger ', ...
        'than options.cgRegularization.']);
end
if ~isempty(cgPreconditioner) && ~isa(cgPreconditioner,'function_handle')
    error('options.cgPreconditioner must be empty or a function handle.');
end
if isempty(mvpPerHvp)
    if any(strcmp(lossType,{'logistic','logistic-regression', ...
            'logistic_regression','lasso','least-squares','least_squares', ...
            'squared-error','squared_error','squarederror'}))
        mvpPerHvp = [1,1];
    else
        mvpPerHvp = [0,0];
    end
elseif ~isnumeric(mvpPerHvp) || numel(mvpPerHvp) ~= 2 || ...
        any(~isfinite(mvpPerHvp)) || any(mvpPerHvp < 0)
    error('options.mvpPerHvp must be [numForward,numTranspose].');
else
    mvpPerHvp = reshape(mvpPerHvp,1,2);
end
end

function cgOperator = resolveCgOperator( ...
        cgOperator,funObj,lossType,numVariables,useCG)
if ~useCG || ~isempty(cgOperator)
    return;
end

[dataMatrix,labels] = objectiveClosureData(funObj,numVariables,lossType);
if any(strcmp(lossType,{'lasso','least-squares','least_squares', ...
        'squared-error','squared_error','squarederror'}))
    cgOperator = @(w,W) squaredErrorCgOperator(dataMatrix,W);
elseif any(strcmp(lossType,{'logistic','logistic-regression', ...
        'logistic_regression'}))
    cgOperator = @(w,W) logisticCgOperator(dataMatrix,labels,w,W);
else
    error(['Cannot build a CG Hessian operator for this objective. ', ...
        'Set options.lossType to ''lasso'' or ''logistic'', or provide ', ...
        'options.cgOperator for a custom objective.']);
end
end

function [dataMatrix,labels] = objectiveClosureData(funObj,numVariables,lossType)
details = functions(funObj);
if ~isfield(details,'workspace') || isempty(details.workspace)
    error(['CG could not recover the data matrix from funObj. Define funObj ', ...
        'as @(w)SquaredError(w,A,b) or @(w)LogisticLoss(w,A,b), or ', ...
        'provide options.cgOperator.']);
end
captured = details.workspace{1};
names = fieldnames(captured);
dataMatrix = [];

preferredMatrixNames = {'A','X'};
for j = 1:numel(preferredMatrixNames)
    name = preferredMatrixNames{j};
    if isfield(captured,name)
        value = captured.(name);
        if isnumeric(value) && ismatrix(value) && size(value,2) == numVariables
            dataMatrix = value;
            break;
        end
    end
end
if isempty(dataMatrix)
    for j = 1:numel(names)
        value = captured.(names{j});
        if isnumeric(value) && ismatrix(value) && ...
                size(value,2) == numVariables && size(value,1) > 1
            dataMatrix = value;
            break;
        end
    end
end
if isempty(dataMatrix)
    error(['CG could not identify the data matrix captured by funObj. ', ...
        'Provide options.cgOperator for this objective.']);
end

labels = [];
if any(strcmp(lossType,{'logistic','logistic-regression', ...
        'logistic_regression'}))
    preferredLabelNames = {'b','y'};
    for j = 1:numel(preferredLabelNames)
        name = preferredLabelNames{j};
        if isfield(captured,name)
            value = captured.(name);
            if isnumeric(value) && isvector(value) && ...
                    numel(value) == size(dataMatrix,1)
                labels = value(:);
                break;
            end
        end
    end
    if isempty(labels)
        for j = 1:numel(names)
            value = captured.(names{j});
            if isnumeric(value) && isvector(value) && ...
                    numel(value) == size(dataMatrix,1)
                labels = value(:);
                break;
            end
        end
    end
    if isempty(labels)
        error(['CG could not identify the logistic labels captured by ', ...
            'funObj. Provide options.cgOperator for this objective.']);
    end
end
end

function op = squaredErrorCgOperator(dataMatrix,W)
workingMatrix = dataMatrix(:,W);
op = @(v) 2*(workingMatrix'*(workingMatrix*v));
end

function op = logisticCgOperator(dataMatrix,labels,w,W)
scores = labels.*(dataMatrix*w);
probability = 1./(1+exp(-scores));
weights = probability.*(1-probability);
workingMatrix = dataMatrix(:,W);
op = @(v) workingMatrix'*(weights.*(workingMatrix*v));
end

function precf = makeCgPreconditioner(factory,w,W,sig)
if isempty(factory)
    precf = [];
else
    precf = factory(w,W,sig);
    if ~isa(precf,'function_handle')
        error('options.cgPreconditioner must return a function handle.');
    end
end
end

%% Psuedo-gradient calculation
function [f,pGrad,smoothGrad] = pseudoGrad(funObj,w,lambda)

[f,smoothGrad] = funObj(w);

f = f + sum(lambda.*abs(w));

pGrad = zeros(size(smoothGrad));
pGrad(smoothGrad < -lambda) = ...
    smoothGrad(smoothGrad < -lambda) + lambda(smoothGrad < -lambda);
pGrad(smoothGrad > lambda) = ...
    smoothGrad(smoothGrad > lambda) - lambda(smoothGrad > lambda);
nonZero = w~=0 | lambda==0;
pGrad(nonZero) = smoothGrad(nonZero) + ...
    lambda(nonZero).*sign(w(nonZero));

end

function residual = computeProxResidual( ...
        w,smoothGrad,lambda,objectiveScale)
scaledGrad = objectiveScale*smoothGrad;
scaledLambda = objectiveScale*lambda;
proxPoint = sign(w-scaledGrad).* ...
    max(abs(w-scaledGrad)-scaledLambda,0);
residual = w-proxPoint;
end

function [w] = orthantProject(w,xi)
w(sign(w) ~= xi) = 0;
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

function out = initializeOutput( ...
        f,funEvals,lossType,mvpPerEval,objectiveScale,w,elapsedTime, ...
        innerSolver,mvpPerHvp,res,stopCriterion)
out = struct();
out.lossType = lossType;
out.mvpPerEval = mvpPerEval;
out.mvpPerHvp = mvpPerHvp;
out.innerSolver = innerSolver;
out.inner_solver = innerSolver;
out.stopCriterion = stopCriterion;
out.objectiveScale = objectiveScale;
out.fvec_native = f;
out.fvec = objectiveScale*f;
out.funEvalsTrace = funEvals;
out.num_Ax = mvpPerEval(1)*funEvals;
out.num_ATy = mvpPerEval(2)*funEvals;
out.mvp = sum(mvpPerEval)*funEvals;
out.timevec = elapsedTime;
out.funEvals = funEvals;
out.numHvp = 0;
out.cg = zeros(0,1);
out.cgInfo = zeros(0,1);
out.cgTol = zeros(0,1);
out.cgMaxIter = zeros(0,1);
out.cgRegularization = zeros(0,1);
out.res = res;
out.iter = 0;
out.outerIter = 0;
out.isAcceptedPoint = true;
out.num_Ax_total = out.num_Ax(end);
out.num_ATy_total = out.num_ATy(end);
out.mvp_total = out.mvp(end);
out.optCond = NaN;
out.nnzvec = nnz(w);
out.nnz = nnz(w);
out.runtime = elapsedTime;
end

function out = appendOutput( ...
        out,f,funEvals,mvpPerEval,objectiveScale,w,iter,stepAccepted, ...
        elapsedTime,numHvp,mvpPerHvp,res)
out.fvec_native(end+1,1) = f;
out.fvec(end+1,1) = objectiveScale*f;
out.funEvalsTrace(end+1,1) = funEvals;
out.num_Ax(end+1,1) = mvpPerEval(1)*funEvals + mvpPerHvp(1)*numHvp;
out.num_ATy(end+1,1) = mvpPerEval(2)*funEvals + mvpPerHvp(2)*numHvp;
out.mvp(end+1,1) = out.num_Ax(end) + out.num_ATy(end);
out.timevec(end+1,1) = elapsedTime;
out.isAcceptedPoint(end+1,1) = stepAccepted;
out.funEvals = funEvals;
out.numHvp = numHvp;
out.res(end+1,1) = res;
out.iter = iter;
out.num_Ax_total = out.num_Ax(end);
out.num_ATy_total = out.num_ATy(end);
out.mvp_total = out.mvp(end);
out.nnzvec(end+1,1) = nnz(w);
out.nnz = nnz(w);
out.runtime = elapsedTime;
end
