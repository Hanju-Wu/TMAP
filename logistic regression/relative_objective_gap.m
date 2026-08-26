function gap = relative_objective_gap(f, f_ref)
%RELATIVE_OBJECTIVE_GAP Normalized objective gap relative to a reference value.
% The max with zero avoids plotting tiny negative gaps caused by roundoff.
if isempty(f_ref)
    gap = NaN(size(f));
    return;
end
gap = max((f - f_ref) ./ max(1, abs(f_ref)), 0);
end
