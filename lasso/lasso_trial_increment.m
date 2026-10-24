function [delta,trialResidual,deltaAx] = lasso_trial_increment(A,x,trialX,residual,mu)
%LASSO_TRIAL_INCREMENT Evaluate a LASSO objective change using one A product.
% residual is the cached Ax-b at the current point, not at a rejected trial.
% The caller may reuse trialResidual for its accepted gradient computation.
step = trialX-x;
deltaAx = A*step;
trialResidual = residual+deltaAx;

% Form coordinatewise L1 changes, avoiding subtraction of two large norms.
changed = find(step);
old = full(x(changed));
next = full(trialX(changed));
displacement = full(step(changed));
absoluteChanges = abs(next)-abs(old);
positive = old>0 & next>=0;
negative = old<0 & next<=0;
absoluteChanges(positive) = displacement(positive);
absoluteChanges(negative) = -displacement(negative);
l1Change = sum(absoluteChanges);
delta = residual'*deltaAx + 0.5*(deltaAx'*deltaAx) + mu*l1Change;
end
