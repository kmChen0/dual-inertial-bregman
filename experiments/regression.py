"""Isolated, predeclared 20-instance preview of the existing regression methods.

No manuscript or source-package files are changed. Original numerical functions
are read and compiled unchanged, except for untimed state snapshots. All figures
and diagnostics are produced here, independently for every generated instance.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'


import sys
sys.dont_write_bytecode=True
import ast
import argparse
import csv
from _core.timing import fcntl
import hashlib
import json
import platform
import types
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'results/generated/regression'
SOURCE=ROOT
DIAG=ROOT/'_core'
METHODS=('BPG','AEPG','ABPG-g','DI-BFB','S-DI-BFB')
DUAL=('DI-BFB','S-DI-BFB')
LABEL={x:x.replace('BFB','BPG') for x in METHODS}
STYLE={'BPG':('#777777',':'),'AEPG':('#009E73','--'),
       'ABPG-g':('#CC79A7','-.'),'DI-BFB':('#0072B2','-'),
       'S-DI-BFB':('#D55E00','--')}
PROTOCOL=dict(seeds=list(range(1,21)),dimensions=[64,256],iterations=3000,
              repeats=3,step_factor=.99,abpg_step_factor=1.,order_seed=8261023,
              fit_relative_gap_window=[1e-10,1e-4],fit_minimum_points=30,
              fit_minimum_dynamic_range=10.,
              band='25th--75th percentiles across independently generated instances',
              timing='serial median of three repeats; diagnostics excluded',
              baseline_best='online incumbent selected by objective values, charged within update time for all methods',
              source_policy='AST-selected original functions; no import-side-effect output creation')
DEPENDENCIES=[DIAG/'orthant.py', DIAG/'weights.py', DIAG/'gain.py',
 DIAG/'baselines.py', DIAG/'regression.py', DIAG/'geometry.py']


def selected_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
    if {x.name for x in nodes}!=set(names):
        raise RuntimeError(f'Missing source functions in {path}')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


common={'np':np,'perf_counter':perf_counter}
orth_ns=selected_functions(DEPENDENCIES[0],('instance','gradient','gap','schedule','step'),common.copy())
orth=types.SimpleNamespace(**orth_ns)
import math
quad_ns=selected_functions(DEPENDENCIES[1],('theta','dexpo','kl','correction'),dict(np=np,math=math))
quad=types.SimpleNamespace(**quad_ns)
simplex=types.SimpleNamespace(**selected_functions(DEPENDENCIES[2],('gain_theta',),dict(np=np)))
old_ns=dict(np=np,perf_counter=perf_counter,quad=quad,orth=orth,simplex=simplex)
selected_functions(DEPENDENCIES[3],('orth_run',),old_ns)
old=types.SimpleNamespace(**old_ns)
publication_ns=dict(np=np,perf_counter=perf_counter,old=old)
selected_functions(DEPENDENCIES[4],('positive_kl','regression_run'),publication_ns)
geometry_ns=dict(np=np,legacy=old)
selected_functions(DEPENDENCIES[5],('dexp','correction','eta','geometry'),geometry_ns)


def instrument(path,name,namespace,marker,snapshot):
    """Only insert an untimed snapshot before the original update timer."""
    tree=ast.parse(path.read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    text=ast.get_source_segment(path.read_text(),node)
    if text.count(marker)!=1:
        raise RuntimeError('Instrumentation anchor is not unique')
    text=text.replace(marker,snapshot+'\n'+marker)
    exec(compile(text,str(path),'exec'),namespace)
    return namespace[name]


old_source=ast.get_source_segment(DEPENDENCIES[3].read_text(),next(
    x for x in ast.parse(DEPENDENCIES[3].read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='orth_run'))
old_source=old_source.replace('rows=[];times=[0.];failure=None;residual=tolerance=0.',
    'rows=[];times=[0.];failure=None;residual=tolerance=0.\n    best_value=float("inf");best_x=x.copy()')
old_source=old_source.replace('        if k==K:break',
    "        if record:\n            rows[-1]['best_accepted_gap']=orth.gap(A,vs,best_x)\n            if method=='DI-BFB':captures.append(dict(u=np.log(x).copy(),s=aux.copy()))\n        if k==K:break")
old_source=old_source.replace('                if not np.all(np.isfinite(x))',
    "                candidate_value=fx if method=='ABPG-g' else objective(x)\n                if method!='ABPG-g':functions+=1\n                if candidate_value<best_value:\n                    best_value=candidate_value;best_x=x.copy()\n                if not np.all(np.isfinite(x))")
exec(compile(old_source,str(DEPENDENCIES[3]),'exec'),old_ns)
captured_old=old_ns['orth_run']
old.orth_run=captured_old
captured_s=instrument(DEPENDENCIES[4],'regression_run',publication_ns,'        times.append(elapsed)',
    "        if record and method=='S-DI-BFB':\n            captures.append(dict(u=u.copy(),s=s.copy(),A=A))")


def read_csv(path):
    with path.open() as f:
        return [{k:float(v) if v else np.nan for k,v in row.items()} for row in csv.DictReader(f)]


def save_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEPENDENCIES}


def run(method,data,count=3000,record=True):
    old_ns['captures']=[];publication_ns['captures']=[]
    result=captured_s(method,data,K=count,record=record,
                      step_factor=1. if method=='ABPG-g' else .99)
    states=publication_ns['captures'] if method=='S-DI-BFB' else old_ns['captures']
    return *result,states


def stable_distance(xs,logz):
    return geometry_ns['dexp'](logz,np.log(xs))


def diagnostics(method,rows,states,data):
    xs=data[2];lam=.99/float(data[0].sum(axis=0).max())
    schedule=orth.schedule(len(rows));best=np.inf
    accepted_before=epoch=epoch_j=0
    maximum_increase=0.;previous_energy=None;previous_epoch=None
    for i,row in enumerate(rows):
        if method=='S-DI-BFB':
            if row['epochs']!=epoch:
                epoch=int(row['epochs']);epoch_j=0
            if row['accepted']>accepted_before:epoch_j+=1
            accepted_before=int(row['accepted'])
            row['epoch']=epoch;row['epoch_j']=epoch_j
            row['accepted_this_call']=int(i>0 and row['accepted']>rows[i-1]['accepted'])
            best=row['best_accepted_gap'] if row['best_accepted_gap']!='' else row['gap']
        else:
            best=row['best_accepted_gap'];row['epoch']=1;row['epoch_j']=i
            row['accepted_this_call']=int(i>0)
        row['best_gap']=float(best);row['relative_gap']=row['gap']/rows[0]['gap']
        row['relative_best_gap']=row['best_gap']/rows[0]['gap']
        row['update_function_count']=row.get('update_function_count',0)
        row['matvec_count']=2*row['gradient_count']+row['update_function_count']
        if method in DUAL:
            state=states[i]
            weight=state.get('A',0. if i==0 else lam/schedule[i-1]**2)
            distance=float(stable_distance(xs,state['s']))
            row['active_weight']=weight;row['A_gap']=weight*row['gap']
            row['comparator_distance']=distance
            row['energy']=row['A_gap']+distance
            if method=='S-DI-BFB' and row['accepted_this_call']:
                if previous_epoch==epoch:
                    maximum_increase=max(maximum_increase,
                        (row['energy']-previous_energy)/max(1.,abs(previous_energy)))
                previous_energy=row['energy'];previous_epoch=epoch
    geo=[];meta={'maximum_energy_increase_scaled':maximum_increase}
    if method=='DI-BFB' and len(rows)>2:
        steps=[]
        for k in range(1,len(rows)):
            p=schedule[k-1];u=states[k-1]['u'];sn=states[k]['s']
            steps.append(dict(u=u,un=states[k]['u'],sn=sn,p=p,Ap=lam/p**2,
                              v=(1-p)*u+p*states[k-1]['s'],oracle_residual=0.))
        geo,audit=geometry_ns['geometry'](steps,rows,xs,lam,False)
        meta.update(audit)
        meta['maximum_reconstruction_residual']=audit['maximum_dual_reconstruction_residual']
        meta.update(final_eta=geo[-1]['eta'],weighted_directional_sum=geo[-1]['weighted_directional_sum'],
                    final_connection_budget=geo[-1]['connection_budget'])
    return rows,geo,meta


def tail_fit(rows,key,window_key='relative_gap'):
    """Predeclared fit on accepted states in the last observed epoch only."""
    last=max(x['epoch'] for x in rows)
    tail=[x for x in rows if x['epoch']==last and x.get('accepted_this_call',1)
          and 1e-10<=x[window_key]<=1e-4 and x[key]>0]
    result=dict(status='insufficient_window',points=len(tail),epoch=last,
                rate=None,r_squared=None,first_j=None,last_j=None)
    if len(tail)<30:return result
    xx=np.array([x['epoch_j'] for x in tail],float)
    yy=np.log([x[key] for x in tail])
    if yy.max()-yy.min()<np.log(10.):
        result['status']='insufficient_dynamic_range'
        return result
    slope,intercept=np.polyfit(xx,yy,1);pred=slope*xx+intercept
    total=float(np.sum((yy-yy.mean())**2))
    result.update(status='fitted' if slope<0 else 'nondecreasing_fit',
                  rate=float(np.exp(slope)),r_squared=float(1-np.sum((yy-pred)**2)/total) if total>0 else None,
                  first_j=int(xx[0]),last_j=int(xx[-1]))
    return result


def step_interpolate(xx,yy,grid):
    """Right-continuous sampled history, without post-horizon carry-forward."""
    xx=np.asarray(xx);yy=np.asarray(yy);grid=np.asarray(grid)
    ix=np.searchsorted(xx,grid,side='right')-1
    out=np.full(grid.shape,np.nan)
    good=(ix>=0)&(grid<=xx[-1]);out[good]=yy[ix[good]]
    return out


def summarize(method,n,seed,rows,clock,failures,meta):
    item=dict(n=n,seed=seed,method=LABEL[method],iterations=int(rows[-1]['k']),
              gradient_count=int(rows[-1]['gradient_count']),
              function_count=int(rows[-1]['update_function_count']),
              matvec_count=int(rows[-1]['matvec_count']),
              backtracking_count=int(rows[-1].get('backtracking_count',0)),
              final_relative_gap=rows[-1]['relative_gap'],
              best_relative_gap=rows[-1]['relative_best_gap'],
              seconds=float(clock[-1]),rejected=int(rows[-1].get('rejected',0)),
              epochs=int(rows[-1].get('epochs',1)),failure_count=sum(x is not None for x in failures))
    if method=='S-DI-BFB':
        reject=[x['k'] for i,x in enumerate(rows[1:],1) if x['rejected']>rows[i-1]['rejected']]
        item['last_rejection_call']=int(reject[-1]) if reject else 0
        item['final_epoch_length']=rows[-1]['epoch_j']
    for target in (1e-4,1e-6,1e-8):
        hit=next((i for i,x in enumerate(rows) if x['relative_best_gap']<=target),None)
        tag=f'{target:.0e}'
        item[f'hit_{tag}']=int(hit is not None)
        item[f'seconds_{tag}']=None if hit is None else float(clock[hit])
        item[f'gradients_{tag}']=None if hit is None else int(rows[hit]['gradient_count'])
    item.update(meta)
    return item


def instance_metadata(data,n,seed):
    matrix,b,xs,vs,tau=data
    values=np.linalg.eigvalsh(matrix.T@(matrix/vs[:,None]))
    return dict(n=n,seed=seed,rank=int(np.linalg.matrix_rank(matrix)),
                hessian_minimum=float(values[0]),hessian_maximum=float(values[-1]),
                min_xstar=float(xs.min()),stationarity=float(np.max(abs(orth.gradient(matrix,b,xs,tau)))),
                L=float(matrix.sum(axis=0).max()),lambda_value=.99/float(matrix.sum(axis=0).max()))


def perform(seeds,count,repeats,tag):
    output=HERE/tag;output.mkdir(exist_ok=True)
    (output/'traces').mkdir(exist_ok=True)
    before=hashes();metadata=[];summaries=[];fits=[];failures=[];started=perf_counter()
    order=np.random.default_rng(PROTOCOL['order_seed'])
    with (HERE/'timing.lock').open('a') as lock:
        print(f'{tag}: waiting for exclusive timing lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(f'{tag}: exclusive lock acquired',flush=True)
        for n in (64,256):
            for seed in seeds:
                data=orth.instance(seed,3*n,n)
                md=instance_metadata(data,n,seed);metadata.append(md)
                np.savez(output/f'instance_n{n}_seed{seed}.npz',A=data[0],b=data[1],xstar=data[2],tau=data[4])
                runs={};clocks={m:[] for m in METHODS};runfail={m:[] for m in METHODS};audits={}
                for method in METHODS:
                    rows,_,failure,states=run(method,data,count,True)
                    rows,geo,audit=diagnostics(method,rows,states,data)
                    runs[method]=rows;audits[method]=audit;runfail[method].append(failure)
                    if geo:save_csv(output/'traces'/f'n{n}_seed{seed}_geometry.csv',geo)
                    if failure:failures.append(dict(n=n,seed=seed,method=method,run='diagnostic',details=failure))
                for repeat in range(repeats):
                    methods=[METHODS[i] for i in order.permutation(len(METHODS))]
                    for method in methods:
                        _,timing,failure,_=run(method,data,count,False)
                        clocks[method].append(timing);runfail[method].append(failure)
                        if failure:failures.append(dict(n=n,seed=seed,method=method,run=f'timing{repeat}',details=failure))
                for method in METHODS:
                    rows=runs[method];length=min(len(rows),*[len(x) for x in clocks[method]])
                    clock=np.median(np.array([x[:length] for x in clocks[method]]),axis=0)
                    rows=rows[:length]
                    for i,row in enumerate(rows):
                        row['median_update_seconds']=float(clock[i])
                    stem=f'n{n}_seed{seed}_{method}'
                    save_csv(output/'traces'/f'{stem}.csv',rows)
                    save_csv(output/'traces'/f'{stem}_timings.csv',[
                        dict(k=i,**{f'repeat{j}':x[i] for j,x in enumerate(clocks[method])})
                        for i in range(length)])
                    summaries.append(summarize(method,n,seed,rows,clock,runfail[method],audits[method]))
                    if method in ('BPG','S-DI-BFB'):
                        keys=['relative_gap']+(['energy'] if method=='S-DI-BFB' else [])
                        for key in keys:
                            fits.append(dict(n=n,seed=seed,method=LABEL[method],quantity=key,
                                             **tail_fit(rows,key)))
                save_csv(output/'per_instance_summary.csv',summaries)
                save_csv(output/'local_contraction_fits.csv',fits)
                dump(output/'failures.json',failures)
                print(f'{tag}: n={n} seed={seed}, elapsed={perf_counter()-started:.1f}s',flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
    after=hashes()
    if after!=before:raise RuntimeError('Source functions changed during run')
    save_csv(output/'instance_metadata.csv',metadata)
    dump(output/'audit.json',dict(protocol={**PROTOCOL,'seeds':list(seeds),'iterations':count,'repeats':repeats},
        elapsed_seconds=perf_counter()-started,source_sha256=before,source_unchanged=True,
        failures=failures,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),all_instances_full_rank=all(x['rank']==x['n'] for x in metadata)))
    return output


def figures(output):
    summary=list(csv.DictReader((output/'per_instance_summary.csv').open()))
    seeds=sorted({int(x['seed']) for x in summary})
    plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    cache={(n,s,m):read_csv(output/'traces'/f'n{n}_seed{s}_{m}.csv')
           for n in (64,256) for s in seeds for m in METHODS}
    for best in (True,False):
        fig,axes=plt.subplots(2,2,figsize=(10.4,7.1),layout='constrained')
        key='relative_best_gap' if best else 'relative_gap'
        for row,n in enumerate((64,256)):
            tmax=min(cache[n,s,m][-1]['median_update_seconds'] for s in seeds for m in METHODS)
            for column,axis in enumerate(('gradient_count','median_update_seconds')):
                grid=np.linspace(0,3000 if column==0 else tmax,501)
                for method in METHODS:
                    curves=[]
                    for seed in seeds:
                        trace=cache[n,seed,method]
                        curves.append(step_interpolate([r[axis] for r in trace],[r[key] for r in trace],grid))
                    matrix=np.array(curves);lo,med,hi=np.nanquantile(matrix,[.25,.5,.75],axis=0)
                    color,ls=STYLE[method];ax=axes[row,column]
                    ax.plot(grid,np.where(med>0,med,np.nan),color=color,ls=ls,label=LABEL[method])
                    good=(lo>0)&(hi>0);ax.fill_between(grid,lo,hi,where=good,color=color,alpha=.12,linewidth=0)
                axes[row,column].set(yscale='log',title=f'n={n}; {len(seeds)} instances',
                    xlabel='Gradient evaluations' if column==0 else 'Update time (s)')
                axes[row,column].grid(alpha=.2)
            axes[row,0].set_ylabel('Relative incumbent gap' if best else 'Relative current-iterate gap')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Median and interquartile range (25–75%); not a confidence interval',fontsize=11)
        name='regression_performance' if best else 'regression_last_iterate'
        for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,4),layout='constrained')
    for ax,n in zip(axes,(64,256)):
        trace=cache[n,1,'S-DI-BFB'];epoch=max(x['epoch'] for x in trace)
        tail=[x for x in trace if x['epoch']==epoch and x['accepted_this_call']]
        for key,label in (('energy',r'$E_j$'),('A_gap',r'$A_j(F(x_j)-F^*)$'),('comparator_distance',r'$D_\omega(x^*,z_j)$')):
            ax.plot([x['epoch_j'] for x in tail],[x[key] if x[key]>0 else np.nan for x in tail],label=label)
        ax.set(yscale='log',xlabel='Accepted iteration in last observed epoch',title=f'n={n}, seed 1')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(output/f'regression_local_energy.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fits=list(csv.DictReader((output/'local_contraction_fits.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,n in zip(axes,(64,256)):
        for j,(method,key,label) in enumerate((('BPG','relative_gap','BPG gap'),('S-DI-BPG','relative_gap','S-DI gap'),('S-DI-BPG','energy','S-DI energy'))):
            values=[float(x['rate']) for x in fits if int(x['n'])==n and x['method']==method and x['quantity']==key and x['status']=='fitted']
            if values:ax.scatter(np.full(len(values),j),values,alpha=.65,s=18)
            ax.text(j,.02,f'{len(values)}/{len(seeds)} fits',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set(xticks=range(3),xticklabels=['BPG gap','S-DI gap','S-DI energy'],ylabel='Empirical per-iteration factor',title=f'n={n}')
        ax.axhline(1,color='gray',lw=.8,ls=':');ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/f'regression_contraction_factors.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.5),layout='constrained')
    for row,n in enumerate((64,256)):
        traces=[read_csv(output/'traces'/f'n{n}_seed{s}_geometry.csv') for s in seeds]
        for col,(key,title) in enumerate((('eta',r'$\eta_k$'),('weighted_directional_sum',r'$\sum\theta\eta$'),('connection_budget',r'$\mathcal{B}_N$'))):
            values=np.array([[x[key] for x in trace] for trace in traces])
            lo,med,hi=np.quantile(values,[.25,.5,.75],axis=0)
            xx=[x['calls'] for x in traces[0]];ax=axes[row,col]
            ax.plot(xx,med,color='#0072B2');ax.fill_between(xx,lo,hi,color='#0072B2',alpha=.15)
            if col==0:ax.set_yscale('symlog',linthresh=1e-9)
            ax.set(title=f'n={n}: '+title,xlabel='DI-BPG gradient evaluations');ax.grid(alpha=.2)
    fig.suptitle('Fresh DI-BPG geometry: median and IQR across instances',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(output/f'regression_geometry.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    aggregate=[]
    for n in (64,256):
        for method in METHODS:
            group=[x for x in summary if int(x['n'])==n and x['method']==LABEL[method]]
            item=dict(n=n,method=LABEL[method],instances=len(group))
            for key in ('final_relative_gap','best_relative_gap','seconds','rejected'):
                lo,med,hi=np.quantile([float(x[key]) for x in group],[.25,.5,.75])
                item.update({f'{key}_q25':lo,f'{key}_median':med,f'{key}_q75':hi})
            for tag in ('1e-04','1e-06','1e-08'):
                hits=[x for x in group if int(x[f'hit_{tag}'])]
                item[f'success_{tag}']=len(hits)
                item[f'median_hit_seconds_{tag}']=float(np.median([float(x[f'seconds_{tag}']) for x in hits])) if hits else ''
            aggregate.append(item)
    save_csv(output/'aggregate_summary.csv',aggregate)
    text=['# Regression multi-instance preview','',f'{len(seeds)} independent seeds per size. Timing uses within-instance medians; repeat counts are in audit.json. Shading is IQR, not a confidence interval.','',
          'Every method selects its incumbent using objective values, with this work charged inside update timing. ABPG-g reuses its already computed candidate objective. Current-iterate curves are provided separately. Timing excludes reference-based diagnostics and plots.','',
          '| n | Method | Median final relative gap | Median best relative gap | Median time (s) | Success at 1e-8 |',
          '|---|---|---:|---:|---:|---:|']
    for x in aggregate:
        text.append(f"| {x['n']} | {x['method']} | {x['final_relative_gap_median']:.3g} | {x['best_relative_gap_median']:.3g} | {x['seconds_median']:.3g} | {x['success_1e-08']}/{len(seeds)} |")
    text+=['','Local fits use accepted points in the last observed epoch with normalized current gap in [1e-10,1e-4], at least 30 points and at least one decade of fitted-quantity variation. Rates and R² are descriptive, not proofs of asymptotic contraction. See local_contraction_fits.csv for unavailable or unsuitable fits.','',
           'Geometry traces were regenerated separately for every DI instance. No seed-1 audit trajectories were reused.']
    (output/'findings.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Additional paired-seed regression benchmark.')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true',help='Run the declared experiment grid.')
    mode.add_argument('--plot-only',type=Path,help='Replot a complete raw-output directory.')
    mode.add_argument('--smoke',action='store_true',help='Short seed-1 check, not a benchmark result.')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,21)))
    parser.add_argument('--iterations',type=int,default=3000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    if args.plot_only:
        figures(args.plot_only)
    else:
        if args.iterations < 2 or args.repeats < 1 or not args.seeds:
            parser.error('Require iterations >= 2 and repeats >= 1.')
        output=args.output.resolve()
        if output.exists() and any(output.iterdir()):
            parser.error('Output directory must be empty; choose --output explicitly to avoid overwriting a run.')
        HERE=output.parent
        HERE.mkdir(parents=True,exist_ok=True)
        out=perform([1] if args.smoke else args.seeds,20 if args.smoke else args.iterations,
                    1 if args.smoke else args.repeats,output.name)
        if not args.smoke and 1 in args.seeds and args.iterations >= 3000:
            figures(out)
        print('Saved results:',out)
