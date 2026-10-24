function[x,tottime,errdecx,lsq, iter,fopt,errVec,timeVec,res,mvp] =...
    Call_FAST_2CDA_ehn(A,b,xsol,tau,fsparsa,varargin)
% Optional 'memory' sets the maximum candidate coordinates per sweep.
% A leading numeric optional argument is accepted as the legacy memory input.

stopcr = 3;
fref = [];
memory = [];
recordFcn = [];
if ~isempty(varargin) && isnumeric(varargin{1})
    memory = varargin{1};
    varargin = varargin(2:end);
end
if mod(numel(varargin),2) ~= 0
    error('FAST:NameValuePairs','Optional parameters must be name-value pairs.');
end
if ~isempty(varargin)
    for j = 1:2:numel(varargin)
        switch lower(varargin{j})
            case 'stopcriterion'
                if strcmpi(varargin{j+1},'relativeFunctionGap')
                    stopcr = 4;
                elseif strcmpi(varargin{j+1},'proxResidual')
                    stopcr = 3;
                else
                    error('Unknown FAST stop criterion.');
                end
            case 'fref'
                fref = varargin{j+1};
            case 'memory'
                memory = varargin{j+1};
            case 'recordfcn'
                recordFcn = varargin{j+1};
            otherwise
                error('FAST:UnknownOption','Unknown FAST option: %s.', ...
                    string(varargin{j}));
        end
    end
end
memoryOptions = {};
if ~isempty(memory)
    % Two-coordinate updates require room for at least one coordinate pair.
    validateattributes(memory,{'numeric'}, ...
        {'real','finite','scalar','integer','>=',2},mfilename,'memory');
    memoryOptions = {'memory',double(memory)};
end
[x,fopt,tottime,iter,errVec,timeVec,res,mvp]=...
    FAST_BCDA(A,b,tau,xsol, 'stopcr', stopcr, 'fref', fref, ...
    'tolstop', fsparsa, 'blvar', 2, 'subop', 1, ...
    'recordFcn',recordFcn,memoryOptions{:});


%error with respect to the real solution
den = xsol'*xsol;
r = xsol - x;
num= r'*r;
errdecx=num/den;

fprintf(1,'Errorx %10.3e\n',errdecx);


%ls error
resid2 = A * x - b;
mvp = mvp + 1;
lsq = 0.5*resid2'*resid2;
fprintf(1,'0.5*||A x - y ||^2 = %10.3e\n',0.5*resid2'*resid2);

fprintf(1,'Number of non-zero components of x = %d\n',...
          sum((abs(x)>=0.0001)));
            
     
fprintf(1,'CPU time so far = %10.3e\n', tottime);
%figure;
%plot(xsol), hold on; plot(x, 'r'), hold on;
%hold off;

end 
