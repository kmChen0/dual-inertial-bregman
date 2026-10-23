"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from time import perf_counter
from . import baselines as legacy

STYLE = {'BPG': ('#777777', ':'), 'ABPG-g': ('#009E73', '-.'),
         'BPPA': ('#777777', ':'), 'DI-BMD': ('#0072B2', '-'),
         'DI-BPPA': ('#0072B2', '-'), 'S-DI-BMD': ('#D55E00', '--'),
         'S-DI-BPPA': ('#D55E00', '--')}


def plot_curve(ax, xx, yy, method):
    yy = np.asarray(yy)
    ax.plot(xx, np.where(yy > 0, yy, np.nan), label=method,
            color=STYLE[method][0], ls=STYLE[method][1])
    ax.set_yscale('log'); ax.grid(alpha=.2)


def expdiv(a, b):
    """D_exp(a,b), stable and finite also for large negative log displacements."""
    delta = a-b
    small = np.abs(delta) < 1e-3
    terms = np.exp(a)-np.exp(b)*(1+delta)
    z = delta[small]
    terms[small] = np.exp(b[small])*z*z*(.5+z*(1/6+z*(1/24+z/120)))
    return float(np.sum(terms))


def nnls_instance(seed, correlation, m=120, n=80):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m,n))
    for j in range(1,n):
        A[:,j] = correlation*A[:,j-1]+np.sqrt(1-correlation**2)*A[:,j]
    A /= np.linalg.norm(A, axis=0)
    truth = np.zeros(n); support = rng.choice(n, n//4, replace=False)
    truth[support] = rng.uniform(.2, 1.2, len(support))
    b = A@truth+.01*rng.normal(size=m)
    xs, _ = nnls(A, b, maxiter=10*n)
    B, c = A.T@A, A.T@b
    gs = B@xs-c
    def gap(x):
        d = x-xs
        return float(.5*np.sum((A@d)**2)+gs@d)
    return A, b, B, c, xs, gs, gap


def nnls_prox(v, warm, B, c, lam, tol=1e-11, cap=100):
    """Damped Newton in log coordinates; no clipping/positivity repair."""
    u = warm.copy(); work = 0
    for _ in range(cap):
        x = np.exp(u); residual = u-v+lam*(B@x-c)
        err = float(np.max(abs(residual)))
        if err <= tol: return u, work, err
        jac = np.eye(len(u))+lam*B*x[None,:]
        du = solve(jac, -residual, assume_a='gen', check_finite=False)
        oldnorm = np.linalg.norm(residual)
        step = 1.
        for _ in range(60):
            un = u+step*du
            if np.max(un) < 700:
                rn = un-v+lam*(B@np.exp(un)-c)
                if np.linalg.norm(rn) <= (1-1e-4*step)*oldnorm:
                    u = un; break
            step *= .5
        else: raise RuntimeError(f'Newton line search failed at {err:g}')
        work += 1
    raise RuntimeError(f'Newton cap reached: {err:g}')


def nnls_run(method, seed, correlation, budget=400, lam=1., tol=1e-11):
    if method not in ('BPPA', 'DI-BPPA', 'S-DI-BPPA', 'V-iEPPA (q=2, heuristic)'):
        raise ValueError('Unsupported method in this reproducibility package')
    Ad, b, B, c, xs, gs, gap = nnls_instance(seed, correlation)
    u = np.full(len(xs), np.log(.2)); s = u.copy(); anchor = u.copy()
    A = 0.; elapsed = 0.; accepted = rejected = epochs = newton = 0
    max_res = margin = guard = 0.; rows = []; lengths = []; best = float('inf')
    selected_value = float('inf'); objective_calls = 0
    vi_c = vi_tau = 1.
    vi = method.startswith('V-iEPPA')
    growing = method == 'G-S-DI-BPPA'
    safeguarded = method in ('S-DI-BPPA', 'C-S-DI-BPPA', 'Soft-S-DI-BPPA', 'G-S-DI-BPPA')
    cumulative = method == 'C-S-DI-BPPA'
    soft = method == 'Soft-S-DI-BPPA'
    recovery_level = soft_accepts = hard_resets = contractions = 0
    retired_weight = retired_gap = accepted_weight = 0.
    soft_margins = soft_slacks = 0.; soft_ledger_errors = []
    credit = raw_credit = np.longdouble(0)
    candidate_credit = debit = 0.; spending_calls = []; ledger_errors = []
    xs_entropy = float(np.sum(xs[xs>0]*np.log(xs[xs>0])-xs[xs>0]))
    def comparator_div(logz):
        return xs_entropy-float(xs@logz)+float(np.exp(logz).sum())
    previous_augmented = None
    ledger_start = comparator_div(anchor)
    initial_comparator_distance = ledger_start
    maximum_edge_residual = 0.
    accepted_margin = comparator_slack = 0.
    maximum_cumulative_residual = 0.
    for call in range(budget+1):
        x = np.exp(u); value = gap(x); best = min(best, value)
        rows.append(dict(k=call, calls=call, gap=value, best_gap=best,
                         update_seconds=elapsed, newton_iterations=newton,
                         accepted=accepted, rejected=rejected, epochs=epochs,
                         objective_calls=objective_calls,
                         active_weight=A, retired_weight=retired_weight,
                         certified_weight=A+retired_weight,
                         soft_accepts=soft_accepts, hard_resets=hard_resets,
                         credit=float(credit), raw_credit=float(raw_credit),
                         candidate_credit=candidate_credit, credit_guard_debit=debit,
                         margin=margin, rounding_guard=guard, max_oracle_residual=max_res,
                         min_x=float(x.min()), minimum_auxiliary_log_coordinate=float(s.min()),
                         auxiliary_below_normal_range=int(np.sum(s < np.log(np.finfo(float).tiny)))))
        if call == budget: break
        trial_recovery = recovery_level
        contracted_weight = 0.
        start = perf_counter()
        old_u, old_s, old_A = u, s, A
        initializing = A == 0.
        parameter = lam*(call+1)**2 if growing and initializing else lam
        th, Ap = (1., parameter) if initializing else legacy.quad.theta(A, lam)
        v = u if method=='BPPA' else anchor if initializing else (1-th)*u+th*s
        if vi:
            if method == 'V-iEPPA (q=1)':
                th = vi_c/(1+vi_c)
            else:
                # Yang--Toh Remark 5.2 practical entropy QSE=2 heuristic.
                th = 2/(1+np.sqrt(1+4*vi_tau/vi_c))
                if vi_tau*th < .1:
                    vi_tau *= 2
                    th = 2/(1+np.sqrt(1+4*vi_tau/vi_c))
            v = np.logaddexp(np.log1p(-th)+u, np.log(th)+s)
        if growing and initializing and parameter > lam:
            # Large first-step parameters amplify double-precision residual roundoff.
            u = u.astype(np.longdouble)
        warm = u.astype(float) if growing and not initializing else u
        un, nit, res = nnls_prox(v, warm, B, c, parameter, tol)
        newton += nit; max_res = max(max_res, res)
        xn = np.exp(un)
        sn = un.copy() if initializing or method=='BPPA' else u+(un-u)/th
        if vi:
            coefficient = 1. if method == 'V-iEPPA (q=1)' else 1/(vi_tau*th)
            sn = s+coefficient*(un-v)
            vi_c *= 1-th
        keep = True
        # Every compared method pays for the same best-point output rule.
        # Incumbent selection uses F, never the reference gap.
        fn = .5*float(np.sum((Ad@xn-b)**2)); objective_calls += 1
        if safeguarded and not initializing:
            gn = B@xn-c; zn = np.exp(sn)
            U = A*.5*float((x-xn)@B@(x-xn))
            terms = Ap*gn*((1-th)*x+th*zn-xn)
            Dzz = expdiv(s, sn)
            margin = float(terms.sum()+Dzz+U)
            guard = 64*np.finfo(float).eps*max(1., np.sum(abs(terms))+abs(Dzz)+abs(U))
            keep = margin >= -guard
            if cumulative:
                candidate_credit = float(credit+np.longdouble(margin)-np.longdouble(guard))
                keep = candidate_credit >= 0
        if cumulative and initializing:
            # BPPA has f=0: C_1=D_omega(p,c)=D_exp(log(c),log(p)).
            raw_credit = np.longdouble(expdiv(anchor, un))
            guard = 64*np.finfo(float).eps*max(1., abs(float(raw_credit)))
            credit = max(np.longdouble(0), raw_credit-np.longdouble(guard))
            debit = guard; candidate_credit = float(credit); margin = float(raw_credit)
        if keep:
            selected_value = min(selected_value,fn)
            if cumulative and not initializing:
                raw_credit += np.longdouble(margin)
                credit += np.longdouble(margin)-np.longdouble(guard)
                debit += guard
                if margin < 0: spending_calls.append(call+1)
            u, s, A = un, sn, Ap; accepted += 1
            accepted_weight += Ap*th
            if soft and trial_recovery==1: soft_accepts += 1
            recovery_level = 0
            if method!='BPPA' and not vi:
                if initializing: epochs += 1; lengths.append(0)
                lengths[-1] += 1
        else:
            rejected += 1
            if soft and trial_recovery==0:
                contracted_weight = A/2
                A /= 2; recovery_level = 1; contractions += 1
            else:
                contracted_weight = A
                anchor, A = s.copy(), 0.; hard_resets += 1
                recovery_level = 2 if soft else 0
            retired_weight += contracted_weight
        elapsed += perf_counter()-start
        if keep and method in ('DI-BPPA', 'S-DI-BPPA', 'G-S-DI-BPPA'):
            # Offline exact-BPPA energy balance; not included in update time.
            if initializing:
                previous_energy = comparator_div(anchor)
                accepted_margin = comparator_slack = 0.
                ledger_start = previous_energy
                edge_margin = expdiv(anchor, un)
            else:
                previous_energy = old_A*gap(np.exp(old_u))+comparator_div(old_s)
                gn = B@xn-c
                edge_margin = float(Ap*gn@((1-th)*np.exp(old_u)+th*np.exp(sn)-xn)
                    +expdiv(old_s, sn)
                    +old_A*.5*((np.exp(old_u)-xn)@B@(np.exp(old_u)-xn)))
            slack = Ap*th*.5*float((xs-xn)@B@(xs-xn))
            energy = Ap*gap(xn)+comparator_div(sn)
            scale = max(1., abs(energy)+abs(previous_energy)+abs(edge_margin)+abs(slack))
            maximum_edge_residual = max(maximum_edge_residual,
                abs(energy-previous_energy+edge_margin+slack)/scale)
            accepted_margin += edge_margin
            comparator_slack += slack
            scale = max(1., abs(energy)+abs(accepted_margin)+abs(comparator_slack)+abs(ledger_start))
            maximum_cumulative_residual = max(maximum_cumulative_residual,
                abs(energy+accepted_margin+comparator_slack-ledger_start)/scale)
        if soft:
            # Offline check of the active/retired-weight energy ledger.
            retired_gap += contracted_weight*value
            if keep:
                soft_margins += expdiv(anchor,un) if initializing else margin
                soft_slacks += Ap*th*.5*float((xs-xn)@B@(xs-xn))
            energy = A*gap(np.exp(u))+comparator_div(s)
            soft_ledger_errors.append(energy+retired_gap+soft_margins+soft_slacks-ledger_start)
        if cumulative and keep:
            # Offline, reference-dependent check; excluded from update timing.
            augmented = A*gap(xn)+comparator_div(s)+float(raw_credit)
            a = Ap if initializing else Ap*th
            slack = a*.5*float((xs-xn)@B@(xs-xn))
            bound = comparator_div(anchor) if initializing else previous_augmented
            ledger_errors.append(augmented+slack-bound)
            previous_augmented = augmented
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(s)):
            raise FloatingPointError('Nonfinite dual iterate')
    projected = np.where(xs > 1e-12, gs, np.minimum(gs, 0))
    return rows, dict(seed=seed, correlation=correlation, m=len(b), n=len(xs),
                      step=lam, tolerance=tol, reference_kkt=float(max(abs(projected))),
                      reference_objective=float(.5*np.sum((Ad@xs-b)**2)),
                      reference_zero_coordinates=int(np.sum(xs==0)),
                      epoch_lengths=lengths, maximum_oracle_residual=max_res,
                      credit_spending_calls=spending_calls,
                      maximum_augmented_ledger_error=max(map(abs,ledger_errors),default=0.),
                      soft_accepts=soft_accepts, hard_resets=hard_resets,
                      weight_contractions=contractions, certified_weight=A+retired_weight,
                      sum_accepted_weight_increments=accepted_weight,
                      maximum_weight_ledger_error=max(map(abs,soft_ledger_errors),default=0.),
                      initial_comparator_divergence=initial_comparator_distance,
                      maximum_normalized_edge_residual=maximum_edge_residual,
                      maximum_normalized_cumulative_residual=maximum_cumulative_residual,
                      final_vi_tau=vi_tau, final_vi_c=vi_c)
