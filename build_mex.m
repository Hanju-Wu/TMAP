function build_mex
%BUILD_MEX Build the three C interfaces used by the main comparisons.
root = setup_paths;
lr = fullfile(root,'logistic regression');
mf = fullfile(root,'lasso','L1GeneralExamples','minFunc_2012');
mex('-largeArrayDims',fullfile(lr,'libsvmread.c'),'-outdir',lr);
mex('-largeArrayDims',fullfile(lr,'IRPN','ripn_cordmin.c'), ...
    '-outdir',fullfile(lr,'IRPN'));
% The upstream lbfgsC gateway declares its dimension array as int.
mex('-compatibleArrayDims',fullfile(mf,'mex','lbfgsC.c'), ...
    '-outdir',fullfile(mf,'compiled'));
rehash;
fprintf('Built libsvmread, ripn_cordmin and lbfgsC for %s.\n',mexext);
end
