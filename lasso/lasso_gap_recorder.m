function [recordFcn,finishFcn] = lasso_gap_recorder(fReference,tolerances)
%LASSO_GAP_RECORDER Scalar trajectory and first-crossing iterate snapshots.
% Solvers call recordFcn(x,f,elapsedTime,mvp,iteration) only for their current
% iterates. No objective/gradient evaluations are performed by the recorder.
validateattributes(fReference,{'numeric'},{'real','finite','scalar'});
validateattributes(tolerances,{'numeric'}, ...
    {'vector','real','positive','finite','nonempty'});
tolerances = tolerances(:);
fvec = zeros(0,1);
timevec = zeros(0,1);
mvpvec = zeros(0,1);
itervec = zeros(0,1);
checkpointIndex = NaN(size(tolerances));
checkpointX = cell(size(tolerances));
recordFcn = @recordPoint;
finishFcn = @finish;

    function recordPoint(x,f,elapsedTime,mvp,iteration)
        index = numel(fvec)+1;
        fvec(index,1) = f;
        timevec(index,1) = elapsedTime;
        mvpvec(index,1) = mvp;
        itervec(index,1) = iteration;
        gap = (f-fReference)/max(1,abs(fReference));
        hits = find(isnan(checkpointIndex) & isfinite(gap) & gap <= tolerances);
        for j = hits(:).'
            checkpointIndex(j) = index;
            checkpointX{j} = x;
        end
    end

    function history = finish()
        history = struct('fReference',fReference,'tolerances',tolerances, ...
            'fvec',fvec,'gapvec',(fvec-fReference)/max(1,abs(fReference)), ...
            'timevec',timevec,'mvpvec',mvpvec,'itervec',itervec, ...
            'checkpointIndex',checkpointIndex,'checkpointX',{checkpointX});
    end
end
