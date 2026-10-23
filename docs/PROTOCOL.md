# Experimental protocol

## KL regression

The objective is `KL(Ax,b) + tau * sum(x)` for nonnegative `x`, with the orthant entropy kernel. Each entry of `A` is initially uniform on `[0,1]`, then each column is normalized to sum to one. The planted positive reference has log-uniform coordinates between 0.1 and 3. We set `tau = 0.1` and `b = exp(tau) * A @ xstar`. Thus the reference is stationary and the gap has the stable identity `F(x)-F(xstar) = KL(Ax,A@xstar)`. No approximate reference solve is needed. All methods start from `x = 0.5`.

The relative-smoothness parameter is the maximum column sum of `A`. BPG, AEPG, DI-BPG and S-DI-BPG use `0.99/L`. ABPG-g uses base `1/L`; its gain starts at 1, is reduced by a factor of 1.5 with lower floor 0.001 before each search, and is multiplied by 1.5 until its majorization test passes. All search trials are charged. AEPG's coordinatewise lower bound is part of that algorithm and is not an added DI safeguard. No baseline receives per-instance outcome-based tuning.

The safeguard uses its computed margin with a scale-dependent floating-point guard (`64 * machine_epsilon * scale`). It is not claimed that floating-point acceptance is identical to exact-arithmetic acceptance when the margin is at roundoff level. Stable KL series are used for small displacements.

The manuscript uses seed 1, 3,000 outer updates, and five serial timing repeats. The additional benchmark uses paired seeds 1–20 and three shuffled serial timing repeats. The shuffle seed is fixed in the runner. All methods share the same instance for each seed. Additional incumbent selection evaluates the original objective within update timing; ABPG-g reuses its candidate objective. This differs from the original current-iterate-only performance protocol and explains why timings from the two protocols should not be combined.

Fresh DI geometry diagnostics record the one-sided third-derivative quantity, accumulated directional bound, signed drift, terminal correction and connection budget. The entropy positive part is used, not the negative part. Stable long-double calculations reduce cancellation where the platform supports extended precision. Diagnostics are excluded from update timing.

## Nonnegative least squares

The objective is `0.5 * ||A x - b||^2` on `x >= 0`, with the orthant entropy kernel for the BPPA family. Matrices have 480 rows and 320 columns. Gaussian columns are correlated recursively with correlation 0.3 or 0.95, then normalized in Euclidean norm. A quarter of the planted signal coordinates are selected uniformly without replacement and assigned uniform values between 0.2 and 1.2; Gaussian noise with standard deviation 0.01 is added to `b`. The noisy optimum's support is not assumed to equal the planted support.

All algorithms start from `x = 0.2`. The BPPA family uses `lambda = 1`, 500 calls, and damped Newton in log coordinates. The stopping criterion is the infinity norm of `u-v + lambda*(A.T@A@exp(u)-A.T@b)`, with tolerance `1e-12` and cap 100 Newton iterations. The Newton line search, linear solves, rejected oracle calls and safeguard tests are timed. There is no clipping or positivity repair of the Bregman iterates. Numerical failure is recorded or raised rather than silently repaired.

V-iBPPA uses the practical entropy exponent-two rule of Yang–Toh, with initial `c = tau = 1` and the stated adaptive doubling rule. It is a heuristic specialization; the code does not assert a global quadratic-scaling property for entropy.

The additional Euclidean baselines are projected FISTA and gradient-restarted FISTA, each with 10,000 steps and step `1/lambda_max(A.T@A)`. Restart tests the sign of `(y-x_next).T @ (x_next-x)`. All methods have access to the shared Gram matrix. FISTA's spectral setup time is reported separately. Objective evaluation for incumbent selection is charged for every method. FISTA iterations are not identified with BPPA oracle calls.

SciPy `optimize.nnls` computes the independent reference with maximum iterations `10*n`. Stable gaps use `0.5*||A(x-xstar)||^2 + gradient_at_xstar.T@(x-xstar)`. Reference data include the KKT residual, active-coordinate counts at three relative thresholds, smallest Gram eigenvalue, and the numerical estimate `||projected_residual||^2/(2*lambda_min)`. This is a floating-point accuracy diagnostic, not an interval-arithmetic certificate. For the seed-1 manuscript references the largest projected KKT infinity residual is approximately `1.60e-14`.

## Recorded results and reproducibility limits

`results/reference/regression` contains aggregate and per-instance summaries, local-fit data, and figures for 20 seeds. `results/reference/nnls` contains the analogous summaries and reference-accuracy statistics; `results/reference/tables` contains the recorded preview tables. These are additional benchmark results, not a declaration that every figure is in the manuscript.

The original timing environment used Python 3.9.12 with NumPy 1.26.3, SciPy 1.11.4 and Matplotlib 3.5.1 on an Apple M1 Pro. Numerical-library threads are set to one before imports by the entry points. The test suite verifies numerical agreement within tolerances, not exact runtime reproduction. Operating system, BLAS, SciPy's NNLS implementation, and the precision of `longdouble` may affect near-roundoff values. macOS was checked locally; Windows locking support is included but was not executed during packaging.

The kernel modules are internal snapshots, not a stable general-purpose solver interface. Some dormant historical options remain in private function bodies to preserve the measured arithmetic. Public benchmark entry points use only the documented fixed-stepsize methods; growing-parameter, cumulative-credit and soft-restart experiments are not part of the reported release protocols.
