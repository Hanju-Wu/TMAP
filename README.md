# Two-metric adaptive projection (TMAP)

MATLAB code for the paper's logistic regression and Gaussian LASSO experiments.
The drivers save objective, elapsed time, and matrix-vector product
trajectories. Use those files to regenerate the tables and figures.

## Build and validate

Use MATLAB R2025b and a supported C compiler. The Gaussian driver uses
Optimization Toolbox (`quadprog`) and Statistics and Machine Learning Toolbox
(`boxplot`). The paper reports timings on an Apple M4 MacBook Air with 16 GB of
RAM, MATLAB R2025b Update 3, and default BLAS and computational-thread settings.

Run the following in MATLAB from the repository root.

```matlab
mex -setup C    % First-time C compiler configuration
build_mex;
```

With the compiler configured, the shell equivalent is
`matlab -batch "build_mex;"`. This builds `lbfgsC` for both L-BFGS variants,
`ripn_cordmin` for IRPN, and `libsvmread` for loading logistic data.
Substitute the full MATLAB executable path if needed.

Run the following checks from the repository root. The second command runs
one Gaussian instance, and all six methods should meet all three targets.
Use it before a full batch or in a separate checkout because the drivers
overwrite output files with matching names.

```sh
matlab -batch "setup_paths; results = runtests(fullfile('lasso','tests','tmapStableTest.m')); assertSuccess(results);"
matlab -batch "setup_paths; set(groot,'defaultFigureVisible','off'); setenv('LASSO_RHO','0.01'); setenv('LASSO_INSTANCES','1'); run(fullfile('lasso','Example.m')); assert(height(metricTable)==18 && all(metricTable.success));"
```

Use the recorded wall-clock times in seconds. They exclude reference solves,
data loading, and Gaussian data generation. The paper uses no additional
feature preprocessing or separate untimed warm-up runs. Record the source
commit, MATLAB version, platform, and any option overrides with each run.

The relative objective gap is `(f - f_ref) / max(1, abs(f_ref))`, where
`f` and `f_ref` are objective values on the same scale. The logistic driver
clips negative gaps to zero. The Gaussian recorder retains signed gaps.

## Logistic regression (Section 5.1)

Place the six LIBSVM inputs in `logistic regression/dataset/`. The required
file names are listed in [DATASETS.md](DATASETS.md).

```sh
matlab -batch "setup_paths; run(fullfile('logistic regression','demo_lr_relative_gap.m'));"
```

### Settings

- All methods start at zero and solve the mean logistic loss with L1 penalty
  `1/m`, where `m` is the sample count. The methods are TMAP-CG, TMAP-L-BFGS,
  PSSgb-L-BFGS, AltN, and IRPN.
- TMAP uses `epsilon=1e-3`, initial step `1`, backtracking factor `0.2`,
  and sufficient-decrease parameter `0.1`. TMAP-CG uses regularization
  coefficient `1e-4`, exponent `0.5`, and CG tolerance `0.1`. CG starts
  at zero without preconditioning. Its adaptive cap of 10–25 iterations
  is relaxed to the working-set dimension when the stopping measure is
  at most 100 times its target.
- Both L-BFGS methods use up to 100 correction pairs. PSSgb uses quadratic
  initialization of the trial step. AltN uses a Barzilai–Borwein step with
  backtracking. IRPN uses coordinate descent for its inner problems.
  The exact options are in [demo_lr_relative_gap.m](logistic%20regression/demo_lr_relative_gap.m).
- Each reference is a TMAP-CG solve to proximal-residual tolerance `1e-10`.
  The driver checks the residual and aborts on a failed reference. The
  comparison runs use a relative objective gap target of `1e-15`.
- The driver supplies cooperative per-solve limits of 25 s for `news20`
  and 200 s for `rcv1.test` to the reference and comparison solvers. The
  other data sets have no time budget. A native call can overrun a budget.

The 25 s and 200 s windows also specify the figure display ranges. The
paper's Table 2 includes measurements from longer trajectories. To collect
a longer table run, use a separate checkout, increase the corresponding
`largeDatasetMaxTime` values (or set them to `Inf`), and report the budgets
used. Keep the paper's display windows when plotting either run.

### Outputs and Table 2

The driver saves `<stem>-relative-gap.mat` and a standalone time-curve PNG
for each data set under `logistic regression/output/`. The stems are
`rcv1_train`, `rcv1_test`, `news20`, `real-sim`, `a9a`, and `ijcnn1`.
Each MAT file contains `labels` and a matching cell array `stats`, plus
`f_ref`, `refOut`, `referenceConverged`, and the options used.

Check `referenceConverged` and extract the first attainment of each target
`q` in `[1e-4, 1e-8, 1e-12]` from the full saved trace for every method.

```matlab
h = stats{j};
k = find(isfinite(h.gapvec) & h.gapvec <= q, 1, 'first');
```

When `k` exists, use `h.timevec(k)` and `h.mvpvec(k)` for time and `N_A`.
Otherwise record an unattained target and note `h.timedOut` if applicable.
A complete Table 2 has 90 data set–method–target entries, including misses.

### Figures 1 and 2

1. Use two rows of three panels. The first row is `rcv1.train`, `news20`,
   and `real-sim`. The second row is `rcv1.test`, `a9a`, and `ijcnn1`.
2. For both figures, keep records with `timevec <= 25` for `news20`,
   `timevec <= 200` for `rcv1.test`, and all points otherwise. Plot in
   recorded order with `timevec` in seconds on the horizontal axis of
   Figure 1 and `mvpvec` on that of Figure 2.
3. Plot `max(gapvec,1e-16)` on a log vertical axis with limits `[1e-16,1]`.
   Apply the floor only when drawing. Mark the last displayed point with
   an `x` for runs whose saved trace never reached `1e-15`. Skip a curve
   if its display window contains no records.
4. Use a shared legend ordered TMAP-CG, TMAP-L-BFGS, PSSgb-L-BFGS, AltN,
   IRPN. Export both figures as vector PDFs.

## Gaussian LASSO (Section 5.2)

The batch driver generates its input data from fixed seeds.

```sh
matlab -batch "setup_paths; set(groot,'defaultFigureVisible','off'); run(fullfile('lasso','run_gaussian_table.m'));"
```

### Settings

- Use `n=2^14`, `m=n/4`, and ten instances per sparsity level
  `rho=0.10, 0.05, 0.01`. Instance `i` uses
  `rng(1000*round(100*rho)+i,'twister')`. The driver draws
  `A=1/sqrt(2*n)*randn(m,n)`, `e=1e-2*randn(m,1)`, and
  `xTrue=sign(sprandn(n,1,floor(rho*m)/n))`, in that order.
  It sets `b=A*xTrue+e` and `gamma=0.1*norm(A'*b,inf)`.
  The objective is `0.5*norm(A*x-b)^2 + gamma*norm(x,1)`.
- For each instance, a TMAP-CG reference solve uses continuation. The driver
  independently verifies the unit-step proximal residual norm against
  `1e-10`. All six comparison methods start at zero at the fixed penalty.
- TMAP uses initial step `1`, `beta=0.2`, `sigma=0.1`, and `epsilon=1e-3`.
  TMAP-CG uses regularization coefficient `0.1` and exponent `0.7`, with the
  norm in the regularization formula capped at `1`. Its CG budget starts at
  five iterations and its tolerance starts at `0.1` and tightens down to
  `1e-7` based on the proximal residual.
- TMAP-L-BFGS stores up to 20 correction pairs and computes the initial
  inverse-Hessian scaling from the latest valid pair on the current working
  set. PSSgb-L-BFGS stores up to 100 pairs and uses sparse trial vectors for
  the forward product `A*x`. It uses the doubled objective internally and
  reports the objective on the scale above.
- The other methods are FAST-2CDA-E, SpaRSA, and ASSN. Each comparison
  method runs once to gap `1e-12` or its iteration or evaluation limit.
  All three table targets come from the same trajectory. The full options
  are in [Example.m](lasso/Example.m) and [tmap.m](lasso/tmap.m).

### Outputs and Table 3

| File in `lasso/output/` | Contents |
| --- | --- |
| `gaussian-n2^14-all-tolerances-summary.csv` | 54 rows indexed by `rho`, `method`, and `tol`. Use `meanTime`, `meanRawMVP`, and `successRate`. |
| `gaussian-n2^14-all-tolerances-instance-metrics.csv` | 540 instance–method–target rows. Use `cpuTime` (elapsed seconds), `rawMVP`, and `success` to verify the means. |
| `gaussian-n2^14-rho-<rho>-trajectories.mat` | One file each for `0.10`, `0.05`, and `0.01`. The cell array `runHistory` is indexed by instance and method, with column names in `method`. The file also stores reference diagnostics. |

Verify `successRate == 1` in each group before reporting a ten-instance
arithmetic mean. Missed targets have `NaN` costs in the per-instance CSV.
Report time to three decimals and mean `N_A` to one decimal. Choose the
shortest mean time per `(rho,tol)` using the unrounded values, then bold
that cell. Use the recorded raw `N_A`, which counts each multiplication
by `A`, a submatrix of `A`, or their transposes once.

### Figures 3 and 4

Use the three MAT files and a common grid of 131 positive gap targets.

```matlab
targets = 10.^((10:-1:-120)'/10);
targets([51 91 131]) = [1e-4 1e-8 1e-12];
```

1. For each sparsity level, method column `j`, target `q`, and instance
   `i`, take `h = runHistory{i,j}`. Use the same first-attainment test
   as in Table 2 on the saved signed gaps, then take time and count at
   that record.
2. Average time and count arithmetically across the ten instances at
   each target. Expect 2,358 rows. Verify ten hits per mean and agreement
   with all 54 summary rows at the three table targets.
3. Make one row of three panels in the order `rho=0.10, 0.05, 0.01`.
   Use mean seconds horizontally in Figure 3 and mean `N_A` in Figure 4.
   Plot the **target** on the log vertical axis with limits `[8e-13,10]`
   and ticks `[1e-12,1e-8,1e-4,1]`. Mark targets `1, 1e-4, 1e-8, 1e-12`
   on each curve.
4. Use a shared legend ordered TMAP-CG, TMAP-L-BFGS, PSSgb-L-BFGS,
   FAST-2CDA-E, SpaRSA, ASSN. Export vector PDFs. The driver's checkpoint
   boxplots are separate from these convergence figures.

## Reproduction checklist

If the requested trajectories already exist, start at the relevant table
or figure subsection.

1. Build the MEX interfaces and run the quick validation.
2. Run the requested full driver. Check reference quality, completion, and
   target attainment before aggregating. Report time limits and misses.
3. Build the requested tables or figures from the saved records. Keep
   generated plotting scripts with the run outputs for reuse.
4. Report the source revision, machine, output paths, and any differences
   from the paper's timings or success counts.
