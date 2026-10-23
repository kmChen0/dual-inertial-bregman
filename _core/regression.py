"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np
import csv
from time import perf_counter
from . import baselines as old
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]/"results/generated/paper"

STYLE = {
    'BPG': ('#777777', ':'), 'AEPG': ('#009E73', '--'),
    'ABPG-g': ('#CC79A7', '-.'), 'DI-BFB': ('#0072B2', '-'),
    'DI-BMD': ('#0072B2', '-'), 'Acc-MD-forward': ('#E69F00', '--'),
    'BPPA': ('#777777', ':'), 'V-iEPPA': ('#CC79A7', '-.'),
    'DI-BPPA': ('#0072B2', '-'), 'S-DI-BPPA': ('#D55E00', '--'),
    'PI-BPG': ('#000000', (0, (5, 2))),
    'PI-BPPA': ('#000000', (0, (5, 2))),
    'S-DI-BFB': ('#D55E00', '--'), 'S-DI-BMD': ('#D55E00', '--')}


def save(name, rows):
    with (OUT / (name + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve(ax, xx, yy, method, label=None):
    yy = np.asarray(yy)
    # Zero gaps are retained in CSV; log-scale lines stop, not floor, at zero.
    ax.plot(xx, np.where(yy > 0, yy, np.nan),
            label=(label or method).replace('DI-BFB', 'DI-BPG'),
            color=STYLE[method][0], ls=STYLE[method][1])
    ax.set_yscale('log')
    ax.grid(alpha=.2)


def positive_kl(x, y):
    """KL(x,y), including its small-displacement series rather than clipping."""
    r = (x-y)/y
    phi = (1+r)*np.log1p(r)-r
    small = np.abs(r) < 1e-4
    z = r[small]
    phi[small] = z*z*(.5+z*(-1/6+z*(1/12+z*(-1/20+z/30))))
    return float(np.sum(y*phi))


def primal_log_center(u, previous, alpha):
    """Log of the unmodified primal center, with an exact positivity test."""
    if alpha == 0:
        return u.copy()
    displacement = previous-u
    if np.any(displacement >= np.log1p(1/alpha)):
        raise FloatingPointError('Primal inertial center leaves the positive orthant')
    return u+np.log1p(-alpha*np.expm1(displacement))


def regression_run(method, data, K=3000, record=True, credit_mode=False, guard=64.,
                   soft_mode=False, step_factor=1.):
    assert not (credit_mode and soft_mode)
    if method not in ('PI-BPG', 'S-DI-BFB'):
        return old.orth_run(method, data, K=K, record=record, step_factor=step_factor)
    matrix, b, xs, vs, tau = data
    lam = step_factor/float(matrix.sum(axis=0).max())
    x = np.full(len(xs), .5)
    u = np.log(x)
    previous, s, anchor = u.copy(), u.copy(), u.copy()
    ax = matrix@x
    A = elapsed = 0.
    gradients = functions = accepted = rejected = epochs = 0
    margin = tolerance = 0.
    credit = raw_credit = np.longdouble(0.)
    credit_trial = initial_credit = error_budget = comparator_slack = 0.
    epoch_distance = accepted_A = 0.
    spent = 0
    recovery = half_accepts = hard_resets = 0
    retired_weight = accepted_weight = retired_gap = accepted_margin = lower_slack = 0.
    initial_distance = old.quad.kl(xs, s) if record else 0.
    rows, times, failure = [], [], None
    best_value, best_x = float('inf'), None
    schedule = old.orth.schedule(K)
    for k in range(K+1):
        if record:
            rows.append(dict(k=k, gap=old.orth.gap(matrix, vs, x),
                             gradient_count=gradients, update_seconds=elapsed,
                             update_function_count=functions, backtracking_count=0,
                             min_x=float(x.min()), accepted=accepted,
                             rejected=rejected, epochs=epochs, margin=margin,
                             roundoff_tolerance=tolerance,
                             energy=A*old.orth.gap(matrix, vs, x)+old.quad.kl(xs, s)
                             if method=='S-DI-BFB' else '',
                             best_accepted_gap=old.orth.gap(matrix, vs, best_x)
                             if best_x is not None else ''))
            if method == 'S-DI-BFB':
                rows[-1].update(active_weight=A, retired_weight=retired_weight,
                                certified_weight=A+retired_weight,
                                sum_accepted_weights=accepted_weight,
                                recovery_level=recovery, half_accepts=half_accepts,
                                hard_resets=hard_resets,
                                weight_ledger_error=rows[-1]['energy']+retired_gap
                                +accepted_margin+lower_slack-initial_distance)
            if credit_mode:
                energy = accepted_A*old.orth.gap(matrix, vs, x)+old.quad.kl(xs, s)
                rows[-1].update(credit=float(credit), raw_credit=float(raw_credit),
                                candidate_credit=credit_trial, initial_credit=initial_credit,
                                error_budget=error_budget, credit_spending_steps=spent,
                                accepted_energy=energy, augmented_energy=energy+float(raw_credit),
                                ledger_error=energy+float(raw_credit)+comparator_slack-epoch_distance
                                if accepted else 0.)
        times.append(elapsed)
        if k == K:
            break
        start = perf_counter()
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                if method == 'PI-BPG':
                    alpha = 0. if k == 0 else schedule[k]*(1-schedule[k-1])/schedule[k-1]
                    v = primal_log_center(u, previous, alpha)
                    w = np.exp(v)
                    un = v-lam*old.orth.gradient(matrix, b, w, tau)
                    gradients += 1
                    xn = np.exp(un)
                    previous, u, x = u.copy(), un, xn
                    accepted += 1
                else:
                    trial_recovery = recovery
                    old_A = A
                    initializing = A == 0.
                    th, Ap = (1., lam) if initializing else old.quad.theta(A, lam)
                    v = anchor if initializing else (1-th)*u+th*s
                    w = np.exp(v)
                    aw = matrix@w
                    g = matrix.T@np.log(aw/b)+tau
                    gradients += 1
                    un = v-lam*g
                    xn = np.exp(un)
                    axn = matrix@xn
                    keep = True
                    if initializing:
                        sn = un.copy()
                        if credit_mode:
                            divergence = positive_kl(xn, w)
                            df = positive_kl(axn, aw)
                            initial_credit = divergence-lam*df
                            tolerance = guard*np.finfo(float).eps*max(1., abs(divergence)+abs(lam*df))
                            raw_credit = np.longdouble(initial_credit)
                            credit = np.longdouble(max(0., initial_credit-tolerance))
                            credit_trial = float(credit)
                            error_budget = tolerance
                    else:
                        sn = (un-(1-th)*u)/th
                        zn = np.exp(sn)
                        U = A*positive_kl(ax, aw)  # h is linear on the interior.
                        df = positive_kl(axn, aw)
                        terms = Ap*g*((1-th)*x+th*zn-xn)
                        Dzz = old.quad.dexpo(s, sn)
                        margin = float(np.sum(terms)+Dzz+U-Ap*df)
                        tolerance = guard*np.finfo(float).eps*max(
                            1., np.sum(abs(terms))+abs(Dzz)+abs(U)+abs(Ap*df))
                        if credit_mode:
                            credit_trial = float(credit+np.longdouble(margin)-np.longdouble(tolerance))
                            keep = credit_trial >= 0.
                        else:
                            keep = margin >= -tolerance
                    # Incumbent selection uses an objective value, not the reference.
                    candidate_value = float(np.sum(axn*np.log(axn/b)+b-axn)+tau*xn.sum())
                    functions += 1
                    if keep:
                        if credit_mode and not initializing:
                            raw_credit += np.longdouble(margin)
                            credit += np.longdouble(margin)-np.longdouble(tolerance)
                            error_budget += tolerance
                            spent += int(margin < -tolerance)
                        if candidate_value < best_value:
                            best_value, best_x = candidate_value, xn.copy()
                        u, s, x, ax, A = un, sn, xn, axn, Ap
                        accepted_A = Ap
                        accepted_weight += Ap*th
                        half_accepts += int(soft_mode and trial_recovery == 1)
                        recovery = 0
                        accepted += 1
                        if initializing:
                            epochs += 1
                    else:
                        if soft_mode and trial_recovery == 0:
                            A *= .5
                            recovery = 1
                        else:
                            anchor, A = s.copy(), 0.
                            recovery = 2 if soft_mode else 0
                            hard_resets += 1
                        retired_weight += old_A-A
                        rejected += 1
                if not np.all(np.isfinite(x)) or np.any(x <= 0):
                    raise FloatingPointError('Nonpositive/nonfinite regression iterate')
        except FloatingPointError as exc:
            elapsed += perf_counter()-start
            failure = dict(iteration=k+1, message=str(exc), elapsed_seconds=elapsed,
                           gradient_count=gradients, accepted=accepted, rejected=rejected)
            break
        elapsed += perf_counter()-start
        if method == 'S-DI-BFB' and record:
            # Reference-dependent checks are offline, not part of the safeguard.
            if keep:
                accepted_margin += (positive_kl(xn, w)-lam*positive_kl(axn, aw)
                                    if initializing else margin)
                lower_slack += Ap*th*positive_kl(vs, aw)
            else:
                retired_gap += (old_A-A)*old.orth.gap(matrix, vs, x)
        if credit_mode and record and keep:
            # Reference-dependent ledger diagnostics are excluded from timings.
            if initializing:
                epoch_distance = old.quad.kl(xs, anchor)
                comparator_slack = lam*positive_kl(vs, aw)
            else:
                comparator_slack += (Ap*th)*positive_kl(vs, aw)
    return rows, times, failure
