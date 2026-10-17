# Rep + peak-effort detector (v2): mediapipe pose (10 fps, largest person) + motion energy.
# Calibrated on CC/PD stock lifts (stock/calibration.json). Steps: shot cuts -> per-shot best periodic landmark
# -> polarity from rest position -> extrema on the excursion side = reps -> peak effort = max speed returning to rest.
import numpy as np
from scipy.signal import find_peaks
CFG=dict(smooth_s=0.3,prom_frac=0.35,min_gap_s=0.9,min_amp=0.08,vis=0.5,min_cov=0.6,cut_k=6.0,cut_min=18.0,min_cycles=2,max_cv=0.4,min_per=0.30,min_shot_s=2.5)  # tuned 2026-10-16, see stock/calibration.json
G={'wrists':[15,16],'hips':[23,24],'shoulders':[11,12],'nose':[0]}
def smooth(x,fps,s):
    n=max(1,int(round(s*fps)));return np.convolve(np.pad(x,(n,n),mode='edge'),np.ones(n)/n,'same')[n:-n]
def cuts(motion,cfg):
    med=np.median(motion)+1e-6;return [i for i in range(1,len(motion)) if motion[i]>max(cfg['cut_k']*med,cfg['cut_min'])]
def body_scale(lm):
    b=np.nanmedian(np.nanmax(lm[:,:,1],1)-np.nanmin(lm[:,:,1],1));return b if np.isfinite(b) and b>0.05 else 1.0
def shot_reps(lm,t,fps,c):
    vis=lm[:,:,2];y=lm[:,:,1].copy();y[vis<c['vis']]=np.nan;bs=body_scale(lm);best=None
    for k,ix in G.items():
        with np.errstate(all='ignore'):s=np.nanmean(y[:,ix],1)
        m=~np.isnan(s);cov=m.mean()
        if cov<c['min_cov'] or m.sum()<5:continue
        s=np.interp(np.arange(len(s)),np.flatnonzero(m),s[m])/bs;x=smooth(s,fps,c['smooth_s'])
        amp=np.percentile(x,95)-np.percentile(x,5)
        if amp<c['min_amp']:continue
        xc=x-x.mean();ac=np.correlate(xc,xc,'full')[len(xc)-1:];ac=ac/(ac[0]+1e-9)
        lo=int(c['min_gap_s']*fps);pk,_=find_peaks(ac[lo:]);per=(ac[lo:][pk].max() if len(pk) else 0)
        sc=amp*cov*max(per,0.05)
        if best is None or sc>best[0]:best=(sc,k,x,amp,per)
    if not best:return None
    _,k,x,amp,per=best
    hist,edges=np.histogram(x,bins=12);rest=edges[np.argmax(hist)]+(edges[1]-edges[0])/2
    pol=1 if (np.percentile(x,95)-rest)>(rest-np.percentile(x,5)) else -1   # +1: excursions go DOWN in image (y up) e.g. squat; -1: go UP (pull-up, press)
    e,_=find_peaks(pol*x,prominence=c['prom_frac']*amp,distance=max(1,int(c['min_gap_s']*fps)))
    if per<c['min_per']:return dict(signal=k,reps=[],peak=[],why=f'aperiodic {per:.2f}',per=float(per))
    if len(e)<c['min_cycles']:return dict(signal=k,reps=[],peak=[],why='<min_cycles',per=float(per))
    d=np.diff(t[e]);cv=float(d.std()/d.mean()) if len(d)>1 else 0
    if cv>c['max_cv']:return dict(signal=k,reps=[],peak=[],why=f'irregular cv={cv:.2f}',per=float(per))
    v=np.gradient(x)*fps*(-pol)   # speed back toward rest = the concentric (effort) phase
    peak=[]
    for i,j in enumerate(e):
        j1=min(len(x)-1,j+int(1.5*fps));peak.append(float(t[j+int(np.argmax(v[j:j1+1]))]) if j1>j else float(t[j]))
    return dict(signal=k,reps=[float(t[j]) for j in e],peak=peak,why='ok',per=float(per),cv=cv,polarity=pol)
def detect(npz,cfg=None):
    c=dict(CFG,**(cfg or {}));d=np.load(npz);t=d['t'];fps=1/np.median(np.diff(t));lm=d['lm'];mo=d['motion']
    cs=[0]+cuts(mo,c)+[len(t)];shots=[];R=[];P=[]
    for a,b in zip(cs[:-1],cs[1:]):
        if (b-a)/fps<c['min_shot_s']:continue
        r=shot_reps(lm[a:b],t[a:b],fps,c)
        if r:shots.append(dict(r,a=float(t[a]),b=float(t[b-1])));R+=r['reps'];P+=r['peak']
    m=smooth(mo,fps,0.3);m[[i for i in cs[1:-1]]]=0;mp,_=find_peaks(m,prominence=1.5,distance=int(0.8*fps))
    return dict(n=len(R),reps=R,peak_effort=P,motion_peaks=[float(x) for x in t[mp]],cuts=[float(t[i]) for i in cs[1:-1]],shots=shots)
