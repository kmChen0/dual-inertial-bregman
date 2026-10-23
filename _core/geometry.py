"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from . import baselines as legacy
HERE = Path(__file__).resolve().parents[1]/"results/generated/paper"

def dexp(a, b):
    a, b = np.asarray(a, dtype=np.longdouble), np.asarray(b, dtype=np.longdouble)
    delta = a-b
    values = np.empty_like(delta)
    moderate = delta <= .5
    values[moderate] = np.exp(b[moderate])*(np.expm1(delta[moderate])-delta[moderate])
    values[~moderate] = np.exp(a[~moderate])-np.exp(b[~moderate])*(1+delta[~moderate])
    small = abs(delta) < 1e-3
    t = delta[small]
    values[small] = np.exp(b[small])*t*t*(.5+t*(1/6+t*(1/24+t*(1/120+t/720))))
    return np.sum(values, dtype=np.longdouble)


def correction(u, s, theta):
    u, s = np.asarray(u, dtype=np.longdouble), np.asarray(s, dtype=np.longdouble)
    theta = np.longdouble(theta)
    return dexp(s, u)-dexp(u+theta*(s-u), u)/theta**2


def eta(u, s):
    d = np.asarray(s, dtype=np.longdouble)-u
    return float(max(0., np.sum(np.exp(np.asarray(s, dtype=np.longdouble))*d**3)))


def check_formulas():
    rng = np.random.default_rng(11)
    errors = []
    for p, q in ((.7, .45), (.1, .08), (.001, .0008)):
        for _ in range(3):
            u, s = rng.normal(size=(2, 8))
            d = s-u
            up = u+p*d
            rho = p+q*(1-p)
            def directional(t):
                return float(np.sum(np.exp(u+t*d)*d**3))
            values = [directional(t) for t in np.linspace(0, 1, 101)]
            assert all(b >= a-1e-12 for a, b in zip(values, values[1:]))
            assert abs(eta(u, s)-max(0., values[-1])) < 1e-10
            gamma = float(correction(u, s, p)-correction(up, s, q))
            first = quad(lambda t: t*(1-p)*(2*p-(1+p)*t)/(2*p*p)*directional(t),
                         0, p, epsabs=1e-12, epsrel=1e-11)[0]
            second = quad(lambda t: (rho-t)**2/(2*q*q)*directional(t),
                          p, rho, epsabs=1e-12, epsrel=1e-11)[0]
            error = abs(gamma-first-second)/max(1., abs(gamma))
            assert error < 1e-9, (p, q, gamma, first+second)
            errors.append(error)
    assert eta(np.zeros(3), np.zeros(3)) == 0
    assert eta(np.zeros(3), -np.ones(3)) == 0
    return max(errors)


def geometry(states, rows, xs, lam, bppa):
    prefix = weighted_sum = cumulative_Q = 0.
    data = []
    gamma_error = reconstruction_error = 0.
    minimum_energy_slack = float('inf')
    positive = xs > 0
    entropy = np.sum(xs[positive].astype(np.longdouble)*(np.log(xs[positive])-1))
    def energy(state, gap):
        s = np.asarray(state['sn'], dtype=np.longdouble)
        return (state['Ap']*gap+entropy-np.sum(xs*s)
                +np.sum(np.exp(s)))
    E1 = float(energy(states[0], rows[1]['gap']))
    for index, state in enumerate(states[1:], start=2):
        p, Ap = state['p'], state['Ap']
        q, _ = legacy.quad.theta(Ap, lam)
        u, un, s = (np.asarray(state[key], dtype=np.longdouble) for key in ('u', 'un', 'sn'))
        d = s-u
        residual = float(np.max(abs(un-u-p*d))/max(1., np.max(abs(un))))
        reconstruction_error = max(reconstruction_error, residual)
        assert residual < 1e-11
        np.testing.assert_allclose(states[index-2]['Ap'], (1-p)*Ap, rtol=2e-12)
        eta_value = eta(u, s)
        r = correction(un, s, q)
        gamma = correction(u, s, p)-r
        endpoint = un+q*(s-un)
        alternative = (-(1-p)/p**2*dexp(un, u)+(1-p)/p*dexp(u, un)
                       +dexp(endpoint, un)/q**2)
        error = float(abs(gamma-alternative)/max(1., abs(gamma), abs(alternative)))
        gamma_error = max(gamma_error, error)
        assert error < 1e-8
        prefix += gamma
        weighted_sum += p*eta_value
        budget = r+prefix
        mass = (1-p)*(p*(2-p)+q*(1-p)**2)/6
        bound_tolerance = 1e-8*max(1., abs(gamma), eta_value)
        assert gamma <= mass*eta_value+bound_tolerance
        assert r <= (1-q)*(1-p)**3*eta_value/6+bound_tolerance
        # The global bound uses eta_N/6, not just the weighted prefix.
        if bppa:
            cumulative_Q += dexp(state['v'], un)/p**2
        E = energy(state, rows[index]['gap'])
        scale = max(1., abs(E1), abs(E), abs(budget), abs(cumulative_Q))
        slack = float((E1+budget-E-cumulative_Q)/scale)
        minimum_energy_slack = min(minimum_energy_slack, slack)
        data.append(dict(calls=index, eta=eta_value, weighted_directional_sum=float(weighted_sum),
                         gamma=float(gamma), terminal_correction=float(r),
                         connection_budget=float(budget), direction_norm=float(np.linalg.norm(d)),
                         cumulative_Q=float(cumulative_Q), energy=float(E),
                         normalized_energy_bound_slack=slack,
                         reconstruction_residual=residual))
    assert minimum_energy_slack > -1e-8, minimum_energy_slack
    return data, dict(maximum_gamma_identity_residual=gamma_error,
                     maximum_dual_reconstruction_residual=reconstruction_error,
                     minimum_normalized_energy_bound_slack=minimum_energy_slack,
                     initial_energy=E1)


def plot(block, cases):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                         'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.1), constrained_layout=True)
    for value, color, style in zip(cases, ('#0072B2', '#D55E00'), ('-', '--')):
        geo, counters, _ = cases[value]
        calls = [r['calls'] for r in geo]
        label = f'$n={value}$' if block == 'regression' else f'$\\rho={value}$'
        eta_values = np.array([r['eta'] for r in geo])
        axes[0].plot(calls, eta_values, color=color, ls=style, lw=1.1, label=label)
        axes[1].plot(calls, [r['weighted_directional_sum'] for r in geo],
                     color=color, ls=style, lw=1.1, label=label)
        axes[2].plot(calls, [r['connection_budget'] for r in geo],
                     color=color, ls=style, lw=1.1, label=label)
        axes[3].step([int(r['gradient_count' if block == 'regression' else 'calls']) for r in counters],
                     [int(r['rejected']) for r in counters], where='post',
                     color=color, ls=style, lw=1.1, label=label)
    # A symmetric log scale displays exact zeros without flooring the data.
    axes[0].set_yscale('symlog', linthresh=1e-6, linscale=.45)
    axes[0].set_ylim(bottom=0)
    axes[0].set_yticks([0, 1e-5, 1e-3, 1e-1, 10])
    axes[0].set_yticklabels(['0', r'$10^{-5}$', r'$10^{-3}$', r'$10^{-1}$', r'$10^1$'])
    method = 'DI-BPG' if block == 'regression' else 'DI-BPPA'
    titles = (f'(a) {method}\n'+r'$\eta_k$',
              f'(b) {method}\n'+r'$\sum_{k=2}^N\theta_{k-1}\eta_k$',
              f'(c) {method}\n'+r'$\mathcal{B}_N$',
              f'(d) S-{method}\nRejected trials')
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel='Gradient evaluations' if block == 'regression' else 'Proximal calls')
        ax.grid(alpha=.2, linewidth=.6)
        assert len(ax.lines) == 2
    axes[0].legend(fontsize=10, handlelength=2.5)
    axes[3].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for suffix in ('png', 'pdf'):
        fig.savefig(HERE/f'{block}_geometry.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
