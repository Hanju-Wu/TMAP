classdef tmapStableTest < matlab.unittest.TestCase
    % Stable LASSO decreases and analytic end-to-end TMAP checks.

    properties (TestParameter)
        innerSolver = struct('CG', 'cg', 'LBFGS', 'lbfgs');
        matrixFormat = struct('dense', 'dense', 'sparse', 'sparse');
    end

    methods (TestClassSetup)
        function addSourceFolder(testCase)
            sourceFolder = fileparts(fileparts(mfilename('fullpath')));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                sourceFolder));
            testCase.assertEqual(which('tmap'), fullfile(sourceFolder, 'tmap.m'));
            testCase.assertEqual(which('lasso_trial_increment'), ...
                fullfile(sourceFolder, 'lasso_trial_increment.m'));
        end
    end

    methods (Test)
        function recoversDecreaseHiddenByObjectiveRounding(testCase)
            % The objective is .5*(x-1)^2 + 2^59. Its decrease from
            % x=0 to its minimizer x=1 is exactly -1/2, independently
            % of the unfit second row. Double objective subtraction loses it.
            A = [1; 0];
            b = [1; -2^30];
            x = 0;
            trialX = 1;
            residual = A*x - b;
            directDelta = 0.5*norm(A*trialX - b)^2 ...
                - 0.5*norm(residual)^2;

            [delta, trialResidual, deltaAx] = ...
                lasso_trial_increment(A, x, trialX, residual, 0);

            testCase.verifyEqual(directDelta, 0, 'AbsTol', 0);
            testCase.verifyEqual(delta, -0.5, 'AbsTol', eps);
            testCase.verifyEqual(trialResidual, [0; 2^30], 'AbsTol', 0);
            testCase.verifyEqual(deltaAx, [1; 0], 'AbsTol', 0);
        end

        function handlesL1SignCrossingsAndZeros(testCase, matrixFormat)
            % Both signs cross, a zero coordinate becomes nonzero, and a
            % nonzero coordinate becomes zero. The squared loss changes
            % from 7 to 110.5 and the one-norm from 6 to 7, so delta=104.25.
            A = matrixInFormat([1 0 1 0; 0 2 0 -1; 1 1 0 1], ...
                matrixFormat);
            x = matrixInFormat([2; -3; 0; 1], matrixFormat);
            trialX = matrixInFormat([-1; 4; -2; 0], matrixFormat);
            residual = matrixInFormat([1; -2; 3], matrixFormat);

            [delta, trialResidual, deltaAx] = ...
                lasso_trial_increment(A, x, trialX, residual, 0.75);

            testCase.verifyEqual(full(delta), 104.25, 'AbsTol', 1e-13);
            testCase.verifyEqual(full(trialResidual), [-4; 13; 6], ...
                'AbsTol', 1e-13);
            testCase.verifyEqual(full(deltaAx), [-5; 15; 3], ...
                'AbsTol', 1e-13);
        end

        function reachesAnalyticScaledIdentitySolution(testCase, innerSolver)
            [A, b, mu, expected] = scaledIdentityProblem();
            opts = solverOptions(innerSolver, numel(expected));

            [x, out] = tmap(A, b, numel(expected), mu, opts);

            verifyAnalyticSolution(testCase, A, b, mu, x, expected, out);
            verifyAcceptedStepsAndCounters(testCase, out);
        end

        function optimalInitialPointNeedsNoTrials(testCase, innerSolver)
            [A, b, mu, expected] = scaledIdentityProblem();
            opts = solverOptions(innerSolver, numel(expected));
            opts.x0 = expected;

            [x, out] = tmap(A, b, numel(expected), mu, opts);

            verifyAnalyticSolution(testCase, A, b, mu, x, expected, out);
            testCase.verifyEqual(out.iter, 0, 'AbsTol', 0);
            testCase.verifyEmpty(out.lineSearchTrials);
            testCase.verifyEmpty(out.lineSearchDelta);
            testCase.verifyEmpty(out.lineSearchRHS);
            testCase.verifyEmpty(out.forcedAcceptance);
            testCase.verifyEqual(out.residualRefreshCount, 0, 'AbsTol', 0);
            testCase.verifyEqual(out.num_Ax_total, 1, 'AbsTol', 0);
            testCase.verifyEqual(out.num_ATy_total, 1, 'AbsTol', 0);
        end

        function refreshingEveryIterationCountsFreshProducts(testCase, innerSolver)
            [A, b, mu, expected] = scaledIdentityProblem();
            opts = solverOptions(innerSolver, numel(expected));
            opts.residualRefreshPeriod = 1;

            [x, out] = tmap(A, b, numel(expected), mu, opts);

            verifyAnalyticSolution(testCase, A, b, mu, x, expected, out);
            verifyAcceptedStepsAndCounters(testCase, out, out.iter);
            testCase.verifyEqual(out.num_Ax_total, ...
                1 + sum(out.lineSearchTrials) + out.nr_CG ...
                + out.residualRefreshCount, 'AbsTol', 0);
            testCase.verifyEqual(out.num_ATy_total, ...
                1 + out.iter + out.nr_CG + out.residualRefreshCount, ...
                'AbsTol', 0);
        end

        function convergesFromNearSolutionDespiteConstantLoss(testCase, innerSolver)
            [smallA, smallB, mu, expected] = scaledIdentityProblem();
            A = [smallA; sparse(1, numel(expected))];
            b = [smallB; 2^30];
            opts = solverOptions(innerSolver, numel(expected));
            opts.x0 = expected + [1; -2; 0; 0; 0; 1.5]*1e-7;
            initialResidual = proxResidual(A, b, mu, opts.x0);
            initialLoss = 0.5*norm(A*opts.x0 - b)^2 ...
                + mu*norm(opts.x0, 1);
            exactLoss = 0.5*norm(A*expected - b)^2 ...
                + mu*norm(expected, 1);

            [x, out] = tmap(A, b, numel(expected), mu, opts);

            testCase.verifyGreaterThan(initialResidual, 1e-8);
            testCase.verifyEqual(initialLoss - exactLoss, 0, 'AbsTol', 0);
            verifyAnalyticSolution(testCase, A, b, mu, x, expected, out);
            verifyAcceptedStepsAndCounters(testCase, out);
        end

        function continuationReachesTargetRegularization(testCase, innerSolver)
            [A, b, ~, ~] = scaledIdentityProblem();
            mu = 0.02;
            expected = softThreshold(2*b, mu)/4;
            opts = solverOptions(innerSolver, numel(expected));
            opts.cont = 1;

            [x, out] = tmap(A, b, numel(expected), mu, opts);

            verifyAnalyticSolution(testCase, A, b, mu, x, expected, out);
            verifyAcceptedStepsAndCounters(testCase, out);
        end

        function continuationReportsTargetReachedOnLastIteration(testCase, innerSolver)
            A = speye(6);
            b = [3; -2; 0.1; -0.08; 0; 1.5];
            mu = 0.3;
            opts = solverOptions(innerSolver, 6);
            opts.cont = 1;
            opts.maxit = 1;
            opts.tol = 1;

            [x, out] = tmap(A, b, 6, mu, opts);
            freshResidual = proxResidual(A, b, mu, x);

            testCase.verifyGreaterThan(0.2*norm(A'*b, 'inf'), mu);
            testCase.verifyLessThanOrEqual(freshResidual, opts.tol);
            testCase.verifyEqual(out.iter, 1, 'AbsTol', 0);
            testCase.verifyEqual(out.msg, 'optimal');
            testCase.verifyEqual(out.res, freshResidual, 'AbsTol', 1e-12);
            verifyAcceptedStepsAndCounters(testCase, out);
        end
    end
end

function A = matrixInFormat(A, matrixFormat)
    if strcmp(matrixFormat, 'sparse')
        A = sparse(A);
    else
        A = full(A);
    end
end

function [A, b, mu, expected] = scaledIdentityProblem()
    A = 2*speye(6);
    b = [3; -2; 0.1; -0.08; 0; 1.5];
    mu = 0.3;
    % Each scalar problem has minimizer soft(2*b_i,mu)/4.
    expected = [1.425; -0.925; 0; 0; 0; 0.675];
end

function opts = solverOptions(innerSolver, n)
    opts = struct('inner_solver', innerSolver, 'x0', zeros(n, 1), ...
        'stopCriterion', 'proxResidual', 'tol', 1e-10, ...
        'maxit', 200, 'cont', 0, 'CG_maxit', n, ...
        'CG_tol', 1e-10, 'CG_adapt', 0);
    % Leave allowForcedAcceptance and residualRefreshPeriod at their defaults.
end

function y = softThreshold(x, mu)
    y = sign(x).*max(abs(x) - mu, 0);
end

function r = proxResidual(A, b, mu, x)
    x = full(x);
    gradient = A'*(A*x - b);
    r = norm(x - softThreshold(x - gradient, mu));
end

function verifyAnalyticSolution(testCase, A, b, mu, x, expected, out)
    testCase.verifyEqual(full(x), expected, 'AbsTol', 1e-9);
    testCase.verifyLessThanOrEqual(proxResidual(A, b, mu, x), 1.1e-10);
    testCase.verifyLessThanOrEqual(out.res, 1.1e-10);
    testCase.verifyEqual(out.msg, 'optimal');
end

function verifyAcceptedStepsAndCounters(testCase, out, expectedRefreshCount)
    if nargin < 3
        expectedRefreshCount = 0;
    end
    testCase.verifyGreaterThan(out.iter, 0);
    testCase.verifySize(out.lineSearchTrials, [out.iter, 1]);
    testCase.verifySize(out.lineSearchDelta, [out.iter, 1]);
    testCase.verifySize(out.lineSearchRHS, [out.iter, 1]);
    testCase.verifyTrue(all(isfinite(out.lineSearchDelta)));
    testCase.verifyTrue(all(isfinite(out.lineSearchRHS)));
    testCase.verifyGreaterThanOrEqual(out.lineSearchTrials, 1);
    testCase.verifyEqual(out.lineSearchTrials, out.backtracks + 1, 'AbsTol', 0);
    testCase.verifyLessThanOrEqual(out.lineSearchDelta, out.lineSearchRHS);
    testCase.verifyLessThanOrEqual(out.lineSearchRHS, 0);
    testCase.verifyEqual(out.forcedAcceptance, false(out.iter, 1));
    testCase.verifyEqual(out.residualRefreshCount, expectedRefreshCount, ...
        'AbsTol', 0);
    testCase.verifyGreaterThanOrEqual(out.num_Ax_total, ...
        1 + sum(out.lineSearchTrials));
    testCase.verifyGreaterThanOrEqual(out.num_ATy_total, 1 + out.iter);
    testCase.verifyGreaterThanOrEqual(diff(out.num_Ax), 0);
    testCase.verifyGreaterThanOrEqual(diff(out.num_ATy), 0);
    testCase.verifyEqual(out.num_Ax_total, out.num_Ax(end), 'AbsTol', 0);
    testCase.verifyEqual(out.num_ATy_total, out.num_ATy(end), 'AbsTol', 0);
end
