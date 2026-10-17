# build_bed.py: full licensed track -> drop-aligned, bar-exact, loopable bed for the montage.
#   run.sh <work> build_bed.py <track.mp3|wav> <out_bed.wav> [--bpm-hint 140] [--out-bpm 140] [--drop-beat N]
# Fits a beat grid (librosa + least squares), finds the drop (biggest post/pre energy + low-end jump), cuts
# [drop-8 beats, drop+24 beats], tempo-stretches to the reel BPM (rubberband, pitch kept), aligns onsets to the
# grid (+-60 ms), and equal-power crossfades the tail into the head so the bed loops. Drop lands at 8 beats (3.43 s @140).
import subprocess,json,sys,os,tempfile,argparse,numpy as np,soundfile as sf,librosa,warnings;warnings.filterwarnings('ignore')
SR=48000;BPM_OUT=140.0;P_OUT=60/BPM_OUT;NB=32;DROP_BEAT=8   # overridden by --out-bpm
def dec(src,dst):subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',src,'-ac','2','-ar',str(SR),'-c:a','pcm_f32le',dst],check=True)
def grid(y,sr,bpm):
    m=y.mean(1);y2=librosa.resample(m,orig_sr=sr,target_sr=22050)
    _,b=librosa.beat.beat_track(y=y2,sr=22050,start_bpm=bpm,tightness=400,units='time')
    P=60/bpm;k=np.round((b-b[0])/P);A=np.vstack([np.ones_like(k),k]).T
    for _ in range(3):
        c,*_=np.linalg.lstsq(A,b,rcond=None);r=b-A@c;keep=np.abs(r)<0.04;A2,b2=A[keep],b[keep];c,*_=np.linalg.lstsq(A2,b2,rcond=None)
    t0,P=c;t0=t0%P
    S=np.abs(librosa.stft(y2,n_fft=1024,hop_length=64));f=librosa.fft_frequencies(sr=22050,n_fft=1024)
    lo=np.maximum(0,np.diff(np.log1p(S[(f>30)&(f<160)].sum(0))));on=librosa.onset.onset_strength(y=y2,sr=22050,hop_length=64);ft=np.arange(len(on))*64/22050
    nb=int((len(y2)/22050-t0)/P)-1;bs=[]
    for d in np.arange(-P/2,P/2,0.002):
        idx=np.clip(np.searchsorted(ft,t0+d+np.arange(nb)*P),0,len(lo)-1);bs.append((on[idx].sum()+0.5*lo[idx].sum(),d))
    t0=t0+max(bs)[1]
    return t0,P,float(np.median(np.abs(r[keep])))
def lowdb(y,sr,a,b):
    s=y[int(a*sr):int(b*sr)].mean(1)
    S=np.abs(np.fft.rfft(s));f=np.fft.rfftfreq(len(s),1/sr);return 10*np.log10((S[(f>30)&(f<160)]**2).sum()+1e-9)-10*np.log10(len(s))
def fulldb(y,sr,a,b):s=y[int(a*sr):int(b*sr)];return 10*np.log10((s**2).mean()+1e-12)
def find_drop(y,sr,t0,P,force=None):
    n=int((len(y)/sr-t0)/P);cands=[]
    for bi in range(DROP_BEAT,n-NB+DROP_BEAT-2):
        t=t0+bi*P
        pre=fulldb(y,sr,t-2*P,t);post=fulldb(y,sr,t,t+4*P);lpre=lowdb(y,sr,t-2*P,t);lpost=lowdb(y,sr,t,t+4*P)
        sustain=fulldb(y,sr,t,t+(NB-DROP_BEAT)*P)
        cands.append((round((post-pre)+0.5*(lpost-lpre),2),bi,round(t,2),round(sustain,1)))
    cands.sort(reverse=True)
    return cands
def build(src,fo,bpm_hint=None,force_beat=None):
    d=tempfile.mkdtemp();wav=os.path.join(d,'src.wav');dec(src,wav);y,sr=sf.read(wav)
    if bpm_hint is None:
        bpm_hint=float(np.atleast_1d(librosa.beat.beat_track(y=librosa.resample(y.mean(1),orig_sr=sr,target_sr=22050),sr=22050)[0])[0])
        while bpm_hint<100:bpm_hint*=2
        while bpm_hint>180:bpm_hint/=2
    t0,P,res=grid(y,sr,bpm_hint);bpm_f=60/P
    c=find_drop(y,sr,t0,P);db=force_beat if force_beat is not None else c[0][1]
    ts=t0+(db-DROP_BEAT)*P;pre=min(1.0,ts);te=t0+(db+NB-DROP_BEAT)*P+0.6
    seg=y[int((ts-pre)*sr):int(te*sr)];sf.write(os.path.join(d,'seg.wav'),seg.astype(np.float32),sr)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',os.path.join(d,'seg.wav'),'-af',f'rubberband=tempo={BPM_OUT/bpm_f:.6f}:pitchq=quality:transients=crisp:detector=percussive','-c:a','pcm_f32le',os.path.join(d,'st.wav')],check=True)
    z,_=sf.read(os.path.join(d,'st.wav'));i0=int(round(pre*bpm_f/BPM_OUT*sr));N=int(round(NB*P_OUT*sr));xf=int(0.08*sr);best=None
    for dd in range(-int(0.06*sr),int(0.06*sr)+1,48):
        w=z[max(0,i0+dd):max(0,i0+dd)+N].mean(1)
        on=librosa.onset.onset_strength(y=librosa.resample(w,orig_sr=sr,target_sr=22050),sr=22050,hop_length=64);ft=np.arange(len(on))*64/22050
        sc=sum(on[np.argmin(np.abs(ft-t))] for t in np.arange(NB)*P_OUT)
        if best is None or sc>best[0]:best=(sc,dd)
    i0+=best[1];mix=z[i0:i0+N+xf].copy();out=mix[:N].copy();u=np.linspace(0,1,xf)[:,None]
    out[:xf]=np.sin(u*np.pi/2)*out[:xf]+np.cos(u*np.pi/2)*mix[N:N+xf];sf.write(fo,out.astype(np.float32),sr)
    info=dict(src=src,bed=fo,src_bpm=round(bpm_f,2),out_bpm=BPM_OUT,stretch_pct=round((BPM_OUT/bpm_f-1)*100,2),drop_src_s=round(t0+db*P,3),drop_in_bed_s=round(DROP_BEAT*P_OUT,3),
              window_src_s=[round(ts,3),round(t0+(db+NB-DROP_BEAT)*P,3)],grid_resid_ms=round(res*1000,1),latency_adj_ms=round(best[1]/sr*1000,1),drop_candidates=[[float(v) for v in x] for x in c[:5]],
              bpm_in_range_130_160=bool(130<=bpm_f<=160))
    json.dump(info,open(fo.rsplit('.',1)[0]+'.bed.json','w'),indent=1);return info
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('track');ap.add_argument('out');ap.add_argument('--bpm-hint',type=float);ap.add_argument('--out-bpm',type=float,default=140.0);ap.add_argument('--drop-beat',type=int)
    a=ap.parse_args();BPM_OUT=a.out_bpm;P_OUT=60/BPM_OUT
    print(json.dumps(build(a.track,a.out,a.bpm_hint,a.drop_beat),indent=1))
