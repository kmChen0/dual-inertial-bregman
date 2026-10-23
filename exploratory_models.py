"""Exploratory exponential/power-law fits on the same predeclared window."""
import argparse
from pathlib import Path
from _core import __doc__ as _runtime_setup
import csv
import numpy as np
ROOT=Path(__file__).resolve().parent
REG=ROOT/"results/generated/regression"

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def local_models():
    """Exploratory equal-parameter fits on the original predeclared window."""
    rows=[]
    for n in (64,256):
        for seed in range(1,21):
            trace=read(REG/'traces'/f'n{n}_seed{seed}_S-DI-BFB.csv')
            epoch=max(float(r['epoch']) for r in trace)
            tail=[r for r in trace if float(r['epoch'])==epoch and float(r['accepted_this_call'])
                  and 1e-10<=float(r['relative_gap'])<=1e-4 and float(r['energy'])>0]
            item=dict(n=n,seed=seed,points=len(tail),status='insufficient_window',
                      exponential_rho=None,exponential_r2=None,power_exponent=None,power_r2=None)
            if len(tail)>=30:
                j=np.array([float(r['epoch_j']) for r in tail]); y=np.log([float(r['energy']) for r in tail])
                if np.ptp(y)>=np.log(10):
                    total=np.sum((y-y.mean())**2)
                    exp=np.polyfit(j,y,1); power=np.polyfit(np.log(j),y,1)
                    item.update(status='fitted',exponential_rho=float(np.exp(exp[0])),
                        exponential_r2=float(1-np.sum((y-np.polyval(exp,j))**2)/total),
                        power_exponent=float(-power[0]),
                        power_r2=float(1-np.sum((y-np.polyval(power,np.log(j)))**2)/total))
                else:item['status']='insufficient_dynamic_range'
            rows.append(item)
    return rows

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=REG)
    parser.add_argument("--output",type=Path,default=ROOT/"results/generated/local_model_comparison.csv")
    args=parser.parse_args()
    REG=args.input
    rows=local_models()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,rows)
    print("Saved exploratory fits:",args.output)
