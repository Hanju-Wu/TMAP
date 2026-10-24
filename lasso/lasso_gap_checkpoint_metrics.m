function T = lasso_gap_checkpoint_metrics(history,A,b,mu,xTrue)
%LASSO_GAP_CHECKPOINT_METRICS Data at the first crossing of each gap target.
% Missing crossings stay NaN. Diagnostic residual/recovery computations use
% saved iterates after the solver finishes and are excluded from run costs.
tol = history.tolerances(:);
cpuTime = NaN(size(tol));
rawMVP = NaN(size(tol));
outerIterations = NaN(size(tol));
proxResidual = NaN(size(tol));
relativeRecoveryError = NaN(size(tol));
relativeFunctionGap = NaN(size(tol));
success = ~isnan(history.checkpointIndex);
for j = find(success(:)).'
    hit = history.checkpointIndex(j);
    w = history.checkpointX{j};
    gradient = A'*(A*w-b);
    proxPoint = sign(w-gradient).*max(abs(w-gradient)-mu,0);
    cpuTime(j) = history.timevec(hit);
    rawMVP(j) = history.mvpvec(hit);
    outerIterations(j) = history.itervec(hit);
    proxResidual(j) = norm(w-proxPoint);
    relativeRecoveryError(j) = norm(w-xTrue)/max(norm(xTrue),eps);
    relativeFunctionGap(j) = history.gapvec(hit);
end
T = table(tol,cpuTime,rawMVP,outerIterations,proxResidual, ...
    relativeRecoveryError,relativeFunctionGap,success);
end
