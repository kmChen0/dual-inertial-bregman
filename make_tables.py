"""Rebuild target-time CSVs and a compact regression LaTeX table from summaries."""
import argparse
import csv
from pathlib import Path
from _core import __doc__ as _runtime_setup
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def target_rows(reference):
    result = []
    for problem, path in (
        ('regression', reference/'regression/aggregate_summary.csv'),
        ('NNLS', reference/'nnls/summary.csv')):
        for row in read(path):
            for tag in ('1e-04', '1e-06', '1e-08'):
                source_tag = '0.0001' if problem == 'NNLS' and tag == '1e-04' else tag
                value = row[f'median_hit_seconds_{source_tag}']
                result.append(dict(
                    problem=problem, setting=row.get('n', row.get('rho')),
                    method=row['method'], relative_target=tag,
                    successful_instances=int(row[f'success_{source_tag}']),
                    instances=int(row['instances']),
                    median_update_hit_ms=1000*float(value) if value else None))
    return result


def regression_latex(reference):
    instances = read(reference/'regression/per_instance_summary.csv')
    lines = [r'\begin{tabular}{llrr}', r'\hline',
             r'$n$ & Method & $10^{-6}$: ms [IQR]; success & $10^{-8}$: ms [IQR]; success \\',
             r'\hline']
    for n in (64, 256):
        for method in ('BPG', 'AEPG', 'ABPG-g', 'DI-BPG', 'S-DI-BPG'):
            group = [r for r in instances if int(r['n']) == n and r['method'] == method]
            cells = []
            for tag in ('1e-06', '1e-08'):
                values = [1000*float(r[f'seconds_{tag}']) for r in group if int(r[f'hit_{tag}'])]
                if values:
                    q25, median, q75 = np.quantile(values, [.25, .5, .75])
                    cells.append(f'{median:.1f} [{q25:.1f}, {q75:.1f}]; {len(values)}/{len(group)}')
                else:
                    cells.append(f'---; 0/{len(group)}')
            lines.append(f'{n} & {method} & '+ ' & '.join(cells)+r' \\')
    lines += [r'\hline', r'\end{tabular}',
              '% Times are medians and IQRs among successful instances; read with success counts.',
              '% Within-instance times are medians of three repeats. This is not a confidence interval.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/reference')
    parser.add_argument('--output', type=Path, default=ROOT/'results/generated/tables')
    args = parser.parse_args()
    rows = target_rows(args.input)
    latex = regression_latex(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'target_times.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'regression_target_times.tex').write_text(latex)
    print('Saved tables:', args.output)


if __name__ == '__main__':
    main()
