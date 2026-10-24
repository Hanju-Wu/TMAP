# Two-metric adaptive projection (TMAP)

MATLAB code for logistic regression and Gaussian reconstruction experiments.

## Requirements

- MATLAB R2025b.
- Optimization Toolbox for FAST-2CDA-E.
- Statistics and Machine Learning Toolbox for boxplots.
- A C compiler supported by MATLAB to build the MEX interfaces.

The included MEX binaries are for Apple Silicon. Rebuild them on other platforms.

## Setup

Open MATLAB in the repository root and run:

```matlab
setup_paths;
mex -setup C       % First-time compiler configuration
build_mex;
```

## Logistic regression

Place the LIBSVM data files in `logistic regression/dataset/`. See [DATASETS.md](DATASETS.md) for file names.

From the repository root:

```matlab
setup_paths;
run(fullfile('logistic regression','demo_lr_relative_gap.m'));
```

Results are saved to `logistic regression/output/`.

## Gaussian reconstruction

No external data files are required.

```matlab
setup_paths;
run(fullfile('lasso','run_gaussian_table.m'));
```

Results are saved to `lasso/output/`. Edit `lasso/run_gaussian_table.m` to change the batch settings.

To run one instance at a single sparsity level:

```matlab
setup_paths;
setenv('LASSO_RHO','0.01');
setenv('LASSO_INSTANCES','1');
run(fullfile('lasso','Example.m'));
setenv('LASSO_RHO','');
setenv('LASSO_INSTANCES','');
```

Rerunning either main experiment overwrites output files with the same names.

## Tests

```matlab
setup_paths;
results = runtests(fullfile('lasso','tests','tmapStableTest.m'));
assertSuccess(results);
```
