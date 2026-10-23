"""Independent NNLS preview; manuscript and archived experiments are read-only.

Protocol: 20 paired instance seeds; 500 exact-BPPA oracle calls, 10,000
projected-FISTA steps; 3 serial timing repeats; median/IQR across instances.
The shared timing lock excludes the concurrent regression benchmark.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ast
import csv
from _core.timing import fcntl
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.linalg import solve
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACK = Path(__file__).resolve().parents[1]
BASE = PACK/"results/generated"
HERE = BASE/"nnls"
SOURCE = PACK/"_core/nnls.py"
METHODS = ("BPPA", "V-iBPPA", "DI-BPPA", "S-DI-BPPA", "FISTA", "Restart-FISTA")
SOURCE_METHOD = {"V-iBPPA": "V-iEPPA (q=2, heuristic)"}
TARGETS = (1e-4, 1e-6, 1e-8)
COLORS = ("#777777", "#CC79A7", "#0072B2", "#D55E00", "#009E73", "#E69F00")


def theta(A, lam):
    t = (math.sqrt(lam) + math.sqrt(lam + 4*A))/2
    return math.sqrt(lam)/t, t*t


def load_source():
    """Read only named legacy functions, not executable module-level imports."""
    tree = ast.parse(SOURCE.read_text())
    chosen = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"expdiv", "nnls_instance", "nnls_prox", "nnls_run"}:
            if node.name == "nnls_run":
                # Preserve the valid trajectory if an oracle raises, without
                # altering any iterate, test, tolerance, or timed operation.
                node.body.insert(0, ast.Global(names=["last_rows"]))
                for i, statement in enumerate(node.body):
                    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "rows" for t in statement.targets):
                        node.body.insert(i+1, ast.Assign(targets=[ast.Name(id="last_rows", ctx=ast.Store())], value=ast.Name(id="rows", ctx=ast.Load())))
                        break
            chosen.append(node)
    env = dict(np=np, solve=solve, nnls=nnls, perf_counter=perf_counter,
               legacy=SimpleNamespace(quad=SimpleNamespace(theta=theta)), last_rows=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[])), str(SOURCE), "exec"), env)
    return env


ENV = load_source()
INSTANCE = ENV["nnls_instance"]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def fista_run(data, budget=10000, restart=False):
    """Standard Euclidean projected FISTA; exact projection, no clipping of gaps.

    All methods pay the same objective-based incumbent selection. Reference
    gaps are evaluated offline, outside update_seconds. FISTA uses the cached
    Gram also available to all BPPA oracles. x^T B x - 2c^T x selects incumbents
    without reference information; original-space objective evaluation is
    performed as in the BPPA comparator for fair output cost.
    """
    Ad, target, B, c, xs, gs, gap = data
    spectral_start = perf_counter()
    L = float(np.linalg.eigvalsh(B)[-1])
    spectral_seconds = perf_counter()-spectral_start
    x = np.full(len(xs), .2)
    y = x.copy()
    t = 1.
    elapsed = 0.
    count_restart = 0
    best = float("inf")
    best_value = float("inf")
    best_x = x.copy()
    rows = []
    for k in range(budget+1):
        value = gap(x)
        best = min(best, value)
        rows.append(dict(k=k, calls=k, gap=value, best_gap=best, update_seconds=elapsed,
                         newton_iterations=0, accepted=k, rejected=0, epochs=1,
                         objective_calls=k, gradient_calls=k, restarts=count_restart,
                         active_weight=0., max_oracle_residual=0., min_x=float(x.min())))
        if k == budget:
            break
        start = perf_counter()
        xn = np.maximum(0., y-(B@y-c)/L)
        fn = .5*float(np.sum((Ad@xn-target)**2))
        if fn < best_value:
            best_value, best_x = fn, xn.copy()
        tn = (1+math.sqrt(1+4*t*t))/2
        yn = xn+(t-1)/tn*(xn-x)
        # O'Donoghue-Candes gradient restart; checked after the normal step.
        if restart and float((y-xn)@(xn-x)) > 0:
            tn = 1.
            yn = xn.copy()
            count_restart += 1
        x, y, t = xn, yn, tn
        elapsed += perf_counter()-start
    return rows, dict(euclidean_L=L, euclidean_L_setup_seconds=spectral_seconds,
                      fista_horizon=budget,
                      final_restarts=count_restart, final_incumbent_objective=best_value)


def reference_stats(seed, rho, data, setup_seconds):
    Ad, b, B, c, xs, gs, gap = data
    scale = max(1., float(np.max(xs)))
    active = xs <= 1e-10*scale
    # Activity thresholds describe sparsity only. A genuine subgradient uses
    # exactly positive coordinates, including arbitrarily small positives.
    projected = np.where(xs > 0, gs, np.minimum(gs, 0.))
    mu = float(np.linalg.eigvalsh(B)[0])
    return dict(seed=seed, rho=rho, m=len(b), n=len(xs),
                active_1e10=int(active.sum()), active_1e8=int(np.sum(xs <= 1e-8*scale)),
                active_1e12=int(np.sum(xs <= 1e-12*scale)),
                positive_coordinates=int((~active).sum()),
                minimum_active_gradient=float(gs[active].min()) if active.any() else None,
                maximum_inactive_gradient=float(np.max(np.abs(gs[~active]))) if (~active).any() else None,
                projected_kkt_inf=float(np.max(np.abs(projected))),
                complementarity_inf=float(np.max(np.abs(xs*gs))),
                reference_objective=.5*float(np.sum((Ad@xs-b)**2)),
                smallest_gram_eigenvalue=mu,
                reference_objective_error_bound=float(projected@projected)/(2*mu),
                euclidean_L=float(np.linalg.eigvalsh(B)[-1]),
                setup_and_reference_seconds=setup_seconds,
                initial_gap=gap(np.full(len(xs), .2)))


def checks():
    diagonal = np.diag([1., 3., 8.])
    c = np.array([1., -2., 4.])
    Ad = np.sqrt(diagonal)
    b = c/np.sqrt(np.diag(diagonal))
    xs = np.maximum(c/np.diag(diagonal), 0)
    gs = diagonal@xs-c
    gap = lambda x: .5*float((x-xs)@diagonal@(x-xs))+float(gs@(x-xs))
    data = Ad, b, diagonal, c, xs, gs, gap
    for restart in (False, True):
        rows, _ = fista_run(data, 1000, restart)
        assert rows[-1]["gap"] < 1e-20
        assert all(r["min_x"] >= 0 for r in rows)
    for rho in (.3, .95):
        data = INSTANCE(1, rho, m=480, n=320)
        saved = np.load(PACK/f"data/paper/nnls/instances_{rho}.npz")
        print("Reference archive keys", list(saved), flush=True)
        # Archive names are checked explicitly rather than inferred by order.
        for key, current in (("A", data[0]), ("b", data[1]), ("xstar", data[4]), ("gradient", data[5])):
            np.testing.assert_allclose(current, saved[key], rtol=2e-11, atol=2e-12)
        ENV["nnls_instance"] = lambda *args: data
        for method in METHODS[:4]:
            source_method = SOURCE_METHOD.get(method, method)
            rows, info = ENV["nnls_run"](source_method, 1, rho, budget=5, tol=1e-12)
            previous = read_csv(PACK/f"data/paper/nnls/rho{rho}_{source_method}_run0.csv")
            np.testing.assert_allclose([r["gap"] for r in rows], [r["gap"] for r in previous[:6]], rtol=1e-10, atol=1e-13)
            assert info["maximum_oracle_residual"] <= 1e-12
        assert reference_stats(1, rho, data, 0)["projected_kkt_inf"] < 1e-10
    result = dict(diagonal_fista=True, archived_seed1_first_five_steps=True,
                  archived_reference_checked=True, strict_oracle_tolerance=1e-12,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    (PACK/"results/generated").mkdir(parents=True,exist_ok=True)
    dump(PACK/"results/generated/nnls_checks.json", result)
    print("PASS", result, flush=True)


def run(seeds, repeats=3, budget=500, fista_budget=10000):
    raw = HERE/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    summaries = json.loads((HERE/"runs.json").read_text()) if (HERE/"runs.json").exists() else []
    stats = json.loads((HERE/"instance_stats.json").read_text()) if (HERE/"instance_stats.json").exists() else []
    indexed = {(r["seed"], r["rho"], r["method"], r["repeat"]): r for r in summaries}
    instance_index = {(r["seed"], r["rho"]): r for r in stats}
    lock = (BASE/"timing.lock").open("a")
    print("WAIT shared timing lock", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("ACQUIRED shared timing lock", flush=True)
    try:
        for rho in (.3, .95):
            for seed in seeds:
                begin = perf_counter()
                data = INSTANCE(seed, rho, m=480, n=320)
                setup = perf_counter()-begin
                ENV["nnls_instance"] = lambda *args: data
                if (seed, rho) not in instance_index:
                    item = reference_stats(seed, rho, data, setup)
                    stats.append(item)
                    instance_index[seed, rho] = item
                    np.savez_compressed(raw/f"instance_rho{rho}_seed{seed}.npz", A=data[0], b=data[1], B=data[2], c=data[3], xs=data[4], gs=data[5])
                for repeat in range(repeats):
                    order = np.random.default_rng(95100+seed*7+repeat).permutation(METHODS)
                    for method in order:
                        key = seed, rho, str(method), repeat
                        if key in indexed:
                            saved = indexed[key]
                            expected = fista_budget if "FISTA" in method else budget
                            if saved["status"] == "completed":
                                assert saved["completed_updates"] == expected, "Cached horizon mismatch"
                            if "FISTA" not in method:
                                assert saved["details"].get("tolerance", 1e-12) == 1e-12, "Cached tolerance mismatch"
                            continue
                        begin = perf_counter()
                        failed = None
                        info = {}
                        try:
                            if "FISTA" in method:
                                rows, info = fista_run(data, fista_budget, method == "Restart-FISTA")
                            else:
                                ENV["last_rows"] = []
                                rows, info = ENV["nnls_run"](SOURCE_METHOD.get(method, method), seed, rho, budget, tol=1e-12)
                        except (RuntimeError, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                            failed = f"{type(exc).__name__}: {exc}"
                            rows = [] if "FISTA" in method else ENV.get("last_rows", [])
                        wall = perf_counter()-begin
                        filename = f"rho{rho}_seed{seed}_{method}_repeat{repeat}.csv"
                        write_csv(raw/filename, rows)
                        result = dict(seed=seed, rho=rho, method=str(method), repeat=repeat,
                                      status="failure" if failed else "completed", failure=failed,
                                      raw_file=filename, complete_wall_seconds=wall, details=info,
                                      completed_updates=int(rows[-1]["calls"]) if rows else 0,
                                      update_seconds=float(rows[-1]["update_seconds"]) if rows else 0.,
                                      final_best_relative_gap=float(rows[-1]["best_gap"]/rows[0]["gap"]) if rows else None,
                                      last_iterate_relative_gap=float(rows[-1]["gap"]/rows[0]["gap"]) if rows else None,
                                      newton_steps=int(rows[-1]["newton_iterations"]) if rows else 0,
                                      rejected=int(rows[-1]["rejected"]) if rows else 0)
                        for target in TARGETS:
                            hit = next((r for r in rows if r["best_gap"] <= target*rows[0]["gap"]), None)
                            result[f"target_{target}_seconds"] = hit["update_seconds"] if hit else None
                            result[f"target_{target}_updates"] = int(hit["calls"]) if hit else None
                        summaries.append(result)
                        indexed[key] = result
                        dump(HERE/"runs.json", summaries)
                        dump(HERE/"instance_stats.json", stats)
                        write_csv(HERE/"instance_stats.csv", stats)
                        print("RUN", seed, rho, repeat, method, "gap", result["final_best_relative_gap"], "time", result["update_seconds"], failed or "OK", flush=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    summarize()


def summarize():
    runs = json.loads((HERE/"runs.json").read_text())
    protocol = json.loads((HERE/'protocol.json').read_text()) if (HERE/'protocol.json').exists() else dict(budget=500,fista_budget=10000,repeats=3)
    grouped = {}
    for row in runs:
        grouped.setdefault((row["rho"], row["seed"], row["method"]), []).append(row)
    per_seed = []
    curves = {}
    for (rho, seed, method), items in sorted(grouped.items()):
        success = [r for r in items if r["status"] == "completed"]
        row = dict(rho=rho, seed=seed, method=method, completed_repeats=len(success),
                   failed_repeats=len(items)-len(success), status="completed" if len(success) == len(items) else "failure")
        for field in ("final_best_relative_gap", "last_iterate_relative_gap", "update_seconds", "newton_steps", "rejected"):
            row[field] = float(np.median([r[field] for r in success])) if success else None
        for target in TARGETS:
            hits = [r[f"target_{target}_seconds"] for r in success]
            row[f"target_{target}_seconds"] = float(np.median(hits)) if hits and all(x is not None for x in hits) else None
        per_seed.append(row)
        if success and len(success) == len(items):
            traces = [read_csv(HERE/"raw"/r["raw_file"]) for r in success]
            length = min(map(len, traces))
            clocks = np.median([[z["update_seconds"] for z in tr[:length]] for tr in traces], axis=0)
            gaps = np.asarray([z["best_gap"]/traces[0][0]["gap"] for z in traces[0][:length]])
            curves[rho, seed, method] = clocks, gaps
    write_csv(HERE/"per_seed_summary.csv", per_seed)
    summary = []
    for rho in (.3, .95):
        for method in METHODS:
            group = [r for r in per_seed if r["rho"] == rho and r["method"] == method]
            if not group:
                continue
            valid = [r for r in group if r["status"] == "completed"]
            row = dict(rho=rho, method=method, instances=len(group), completed=len(valid), failures=len(group)-len(valid))
            for field in ("final_best_relative_gap", "last_iterate_relative_gap", "update_seconds", "newton_steps", "rejected"):
                vals = [r[field] for r in valid]
                for label, q in (("q25", .25), ("median", .5), ("q75", .75)):
                    row[f"{field}_{label}"] = float(np.quantile(vals, q)) if vals else None
            for target in TARGETS:
                hits = [r[f"target_{target}_seconds"] for r in valid if r[f"target_{target}_seconds"] is not None]
                row[f"success_{target}"] = len(hits)
                row[f"median_hit_seconds_{target}"] = float(np.median(hits)) if hits else None
            summary.append(row)
    write_csv(HERE/"summary.csv", summary)
    dump(HERE/"summary.json", summary)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for ri, rho in enumerate((.3, .95)):
        sets = {method: [curves[r, seed, m] for r, seed, m in curves if r == rho and m == method] for method in METHODS}
        # Each method stops at the shortest observed horizon across its own
        # completed instances. Never carry terminal values forward in time.
        all_curves = [c for seq in sets.values() for c in seq]
        if not all_curves:
            continue
        for method, color in zip(METHODS, COLORS):
            available = sets[method]
            if not available:
                continue
            horizon = min(c[0][-1] for c in available)
            time_grid = np.linspace(0, horizon, 350)
            grids = []
            for times, gaps in available:
                idx = np.searchsorted(times, time_grid, side="right")-1
                grids.append(gaps[np.maximum(idx, 0)])
            lo, mid, hi = np.quantile(grids, [.25, .5, .75], axis=0)
            axes[ri, 0].plot(time_grid, np.where(mid > 0, mid, np.nan), color=color, label=method)
            axes[ri, 0].fill_between(time_grid, np.where(lo > 0, lo, np.nan), np.where(hi > 0, hi, np.nan), color=color, alpha=.13)
            if "FISTA" not in method:
                min_n = min(len(c[1]) for c in available)
                lo, mid, hi = np.quantile([c[1][:min_n] for c in available], [.25, .5, .75], axis=0)
                axes[ri, 1].plot(np.arange(min_n), np.where(mid > 0, mid, np.nan), color=color, label=method)
                axes[ri, 1].fill_between(np.arange(min_n), np.where(lo > 0, lo, np.nan), np.where(hi > 0, hi, np.nan), color=color, alpha=.13)
        for col in range(2):
            axes[ri, col].set_yscale("log")
            axes[ri, col].set_ylim(1e-12, 1.05)
            axes[ri, col].grid(alpha=.2)
            axes[ri, col].set_ylabel("Best accepted relative objective gap")
            axes[ri, col].set_title(rf"NNLS: $m=480$, $n=320$, $\rho={rho}$")
        axes[ri, 0].set_xlabel("Update time (s; within-instance median)")
        axes[ri, 1].set_xlabel("Bregman proximal calls (including rejections)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(HERE/f"nnls_multiseed.{ext}", dpi=220)
    plt.close(fig)
    findings = ["# NNLS multi-seed preview", "", f"Protocol: paired seeds; m=480,n=320; rho=0.3,0.95; BPPA-family horizon {protocol['budget']} calls; FISTA horizon {protocol['fista_budget']} gradient steps; Newton residual <=1e-12. {protocol['repeats']} serial timing repeats, single-thread BLAS, shared lock. Runtime excludes reference diagnostics and includes the same original-space objective-based incumbent selection for every method. Instance preprocessing/reference costs are separate.", "", "V-iBPPA denotes the archived Yang-Toh practical entropy exponent-2 heuristic, not a theorem-certified exponent-2 entropy complexity claim. Restart-FISTA uses the gradient restart test. Shading: instancewise 25th-75th percentiles, not confidence intervals. FISTA step counts are not equated to exact-proximal calls. Each method stops at its common observed time interval across instances, with no terminal-value carry-forward. Signed gaps at reference-roundoff resolution can be slightly negative: these are retained in raw files, omitted from logarithmic plots, and are not evidence of negative true objective gaps. Reference accuracy diagnostics are in instance_stats.csv.", "", "Failures are retained in runs.json and summary.csv; plots aggregate instances whose timing repeats all completed, so interpret alongside completion counts. Target success denominator is all requested instances.", "", "|rho|method|completed|median final best relative gap|median update seconds|target 1e-8 successes|", "|---|---|---|---|---|---|"]
    for r in summary:
        fmt = lambda x: "NA" if x is None else f"{x:.4g}"
        findings.append(f"|{r['rho']}|{r['method']}|{r['completed']}/{r['instances']}|{fmt(r['final_best_relative_gap_median'])}|{fmt(r['update_seconds_median'])}|{r['success_1e-08']}/{r['instances']}|")
    (HERE/"findings.md").write_text("\n".join(findings)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--summarize',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,21)))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--budget',type=int,default=500)
    parser.add_argument('--fista-budget',type=int,default=10000)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    HERE=args.output.resolve()
    BASE=HERE.parent
    if args.check:
        checks()
    elif args.summarize:
        summarize()
    else:
        if args.budget < 1 or args.fista_budget < 1 or args.repeats < 1:
            parser.error('Budgets and repeats must be positive.')
        if HERE.exists() and any(HERE.iterdir()):
            parser.error('Use a new empty --output directory; reruns do not silently reuse cached timings.')
        HERE.mkdir(parents=True,exist_ok=True)
        seeds=[1] if args.smoke else args.seeds
        repeats=1 if args.smoke else args.repeats
        budget=5 if args.smoke else args.budget
        fista_budget=10 if args.smoke else args.fista_budget
        dump(HERE/'protocol.json',dict(seeds=seeds,repeats=repeats,budget=budget,
            fista_budget=fista_budget,oracle_tolerance=1e-12,smoke=args.smoke,
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            platform=platform.platform(),numerical_library_threads=1))
        run(seeds,repeats,budget,fista_budget)
