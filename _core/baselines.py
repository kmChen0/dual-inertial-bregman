"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np
from time import perf_counter
from . import orthant as orth, weights as quad, gain as simplex

def orth_run(method,data,K=3000,scale=1.,record=True,step_factor=1.):
    A,b,xs,vs,tau=data;lam=step_factor/float(A.sum(axis=0).max())
    x=np.full(len(xs),.5*scale)
    aux=np.log(x) if method=='DI-BFB' else x.copy()
    logz=np.log(x);pg=pt=1.;gain=th=1.
    elapsed=0.;gradients=proxes=functions=backtracks=0
    rows=[];times=[0.];failure=None;residual=tolerance=0.
    theta=orth.schedule(K)
    def objective(z):
        v=A@z
        return float(np.sum(v*np.log(v/b)+b-v)+tau*z.sum())
    for k in range(K+1):
        if record:
            logs=aux if method=='DI-BFB' else np.log(aux[aux>0])
            rows.append(dict(k=k,gap=orth.gap(A,vs,x),update_seconds=elapsed,
                gradient_count=gradients,prox_count=proxes,matvec_count=2*gradients+functions,
                update_function_count=functions,backtracking_count=backtracks,
                active_fraction=float(np.mean(aux==0)) if method=='AEPG' else 0.,
                min_x=float(x.min()),log_aux_min=float(logs.min()),log_aux_max=float(logs.max()),
                gain=gain,theta=th,majorization_residual=residual,roundoff_tolerance=tolerance))
        if k==K:break
        start=perf_counter()
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                if method!='ABPG-g':
                    th=theta[k]
                    x,aux,_=orth.step(method,A,b,tau,x,aux,th,lam)
                    gradients+=1;proxes+=1
                else:
                    gain=max(pg/1.5,.001)
                    for trial in range(100):
                        th=1. if k==0 else simplex.gain_theta(gain,pg,pt)
                        y=(1-th)*x+th*aux;gy=orth.gradient(A,b,y,tau)
                        logzn=logz-lam/(gain*th)*gy
                        zn=np.exp(logzn);xn=(1-th)*x+th*zn
                        gradients+=1;proxes+=1
                        fy,fx=objective(y),objective(xn);functions+=2
                        kl=quad.dexpo(logz,logzn)
                        linear=float(gy@(xn-y));bound=gain*th**2/lam*kl
                        residual=fx-fy-linear-bound
                        tolerance=32*np.finfo(float).eps*max(1.,abs(fx),abs(fy),abs(linear),abs(bound))
                        if residual<=tolerance:break
                        gain*=1.5;backtracks+=1
                    else:raise RuntimeError('ABPG-g gain cap exceeded')
                    x,aux,logz=xn,zn,logzn;pg,pt=gain,th
                if not np.all(np.isfinite(x)) or np.any(x<=0) or not np.all(np.isfinite(aux)):
                    raise FloatingPointError('Nonpositive/nonfinite state')
        except (FloatingPointError,RuntimeError) as exc:
            elapsed+=perf_counter()-start
            failure=dict(iteration=k+1,message=str(exc),elapsed_seconds=elapsed,
                         gradient_count=gradients,prox_count=proxes,
                         update_function_count=functions,backtracking_count=backtracks)
            break
        elapsed+=perf_counter()-start
        times.append(elapsed)
    return rows,times,failure
