# Pad/strike-sound detector: decides whether to OFFER fight mode (one confirm). 16 kHz mono wav, 10 s windows.
import numpy as np,soundfile as sf
CFG=dict(hop=160,flux_db=8.0,min_ioi=0.08,win=10.0,hf_hz=2000)
def onsets(x,sr,c):
    h=c['hop'];n=len(x)//h;e=np.sqrt((x[:n*h].reshape(n,h)**2).mean(1)+1e-12);le=20*np.log10(e)
    fl=np.maximum(0,np.diff(le,prepend=le[0]));base=np.convolve(le,np.ones(50)/50,'same')
    on=[i for i in range(2,n-10) if fl[i]>=c['flux_db'] and fl[i]>=fl[i-2:i+3].max() and le[i]>base[i]+3]
    out=[]
    for i in on:
        if out and (i-out[-1])*h/sr<c['min_ioi']:continue
        out.append(i)
    return np.array(out),le
def features(x,sr,c=CFG):
    on,le=onsets(x,sr,c);h=c['hop'];dur=len(x)/sr
    if len(on)==0:return dict(rate=0,sharp=0,hf=0,combo=0,n=0)
    sharp=[];hf=[]
    for i in on:
        pk=le[i:i+3].max();d=le[i:i+9]                        # decay within 80 ms by >=12 dB
        sharp.append(float((pk-d.min())>=12))
        seg=x[i*h:i*h+int(0.03*sr)]
        if len(seg)>64:
            S=np.abs(np.fft.rfft(seg*np.hanning(len(seg))))**2;f=np.fft.rfftfreq(len(seg),1/sr);hf.append(S[f>=c['hf_hz']].sum()/(S.sum()+1e-12))
    ioi=np.diff(on)*h/sr;combo=float(np.mean((ioi>0.12)&(ioi<0.6))) if len(ioi) else 0
    return dict(rate=len(on)/dur*10,sharp=float(np.mean(sharp)),hf=float(np.mean(hf)) if hf else 0,combo=combo,n=len(on))
def windows(path,c=CFG):
    x,sr=sf.read(path,dtype='float32');x=x if x.ndim==1 else x.mean(1);W=int(c['win']*sr);out=[]
    for a in range(0,max(1,len(x)-W+1),W):
        if a+W<=len(x) or a==0:out.append(features(x[a:a+W],sr,c))
    return out
# ---- calibrated decision (calibrated on 22 labelled test clips): offer fight mode ONCE per source if
#  audio:  >=40% of 10 s windows have onset rate >=7/10 s AND sharp-decay fraction >=0.45 AND HF(>2 kHz) ratio <=0.12
#  or visual+audio: two similar-size people in >=25% of 1 fps frames AND median onset rate >=7 AND combo (IOI 0.12-0.6 s) >=0.4
DEC=dict(rate=7.0,sharp=0.45,hf=0.12,frac=0.4,two_person=0.25,combo=0.4)
def offer_fight(wav,video=None,d=DEC):
    W=windows(wav);wp=[w['rate']>=d['rate'] and w['sharp']>=d['sharp'] and w['hf']<=d['hf'] for w in W];frac=float(np.mean(wp)) if W else 0
    med=lambda k:float(np.median([w[k] for w in W])) if W else 0
    tp=None
    if frac<d['frac'] and video and med('rate')>=d['rate'] and med('combo')>=d['combo']:
        from two_person import two_person;tp=two_person(video)
    yes=frac>=d['frac'] or (tp is not None and tp>=d['two_person'])
    return dict(offer=bool(yes),audio_window_frac=round(frac,2),two_person=tp,rule='audio' if frac>=d['frac'] else ('visual+audio' if yes else 'none'))
