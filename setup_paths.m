function root = setup_paths
%SETUP_PATHS Add only the directories used by the paper experiments.
root = fileparts(mfilename('fullpath'));
lr = fullfile(root,'logistic regression');
ls = fullfile(root,'lasso');
vendor = fullfile(ls,'L1GeneralExamples');
addpath(root,lr,fullfile(lr,'IRPN'),ls,fullfile(ls,'FAST-BCDA'), ...
    fullfile(vendor,'L1General2_2010'),fullfile(vendor,'lossFuncs'), ...
    fullfile(vendor,'misc'),fullfile(vendor,'minFunc_2012'), ...
    fullfile(vendor,'minFunc_2012','compiled'));
end
