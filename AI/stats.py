import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm

def fisher_z(r):  # handle numeric and NaN
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5*np.log((1+r)/(1-r))

def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2: return np.nan, np.nan
    return ttest_ind(a, b, equal_var=False)

def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    m = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan, float)
    c = float(m)
    prev = 1.0
    for i, idx in enumerate(order):
        if not np.isfinite(p[idx]): continue
        rank = i+1
        val = p[idx]*c/rank
        prev = min(prev, val)
        q[idx] = prev
    return q

def sign_kappa_like(vals, mean_val):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(mean_val) or mean_val == 0: return np.nan
    return (np.sign(vals) == np.sign(mean_val)).mean()

def dl_random_effects(effects, ses):
    effects = np.asarray(effects, float); ses = np.asarray(ses, float)
    mask = np.isfinite(effects) & np.isfinite(ses) & (ses>0)
    effects, ses = effects[mask], ses[mask]
    k = effects.size
    if k == 0: return dict(k=0, mean=np.nan, se=np.nan, ci=(np.nan,np.nan), Q=np.nan, I2=np.nan, tau2=np.nan)
    w = 1/(ses**2); mu = np.sum(w*effects)/np.sum(w)
    Q = np.sum(w*(effects-mu)**2); df = k-1
    C = np.sum(w) - (np.sum(w**2)/np.sum(w))
    tau2 = max(0.0, (Q-df)/C) if df>0 else 0.0
    w_re = 1/(ses**2 + tau2)
    mu_re = np.sum(w_re*effects)/np.sum(w_re)
    se_re = (1/np.sum(w_re))**0.5
    z = norm.ppf(0.975)
    I2 = max(0.0, (Q-df)/Q)*100.0 if (Q>0 and df>0) else 0.0
    return dict(k=k, mean=mu_re, se=se_re, ci=(mu_re-z*se_re, mu_re+z*se_re), Q=Q, I2=I2, tau2=tau2)
