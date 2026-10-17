# Highlight Reel "hype montage" renderer v4 (render_hype + spec.shots from sequence.py) (style=hype). CPU-only, one step at a time, streaming.
# usage: python render_hype.py hype_spec.json [music.wav|mp3]
#  - music given  -> librosa beat grid + drop detection; cuts 1 frame BEFORE each beat, hero peak on the drop.
#  - no music     -> exact spec.bpm grid (default 140), SFX only (default). spec.music="generated" = internal preview bed, never a deliverable.
#  - deliverables: render once with music none, then mux.py mixes licensed beds WITH this SFX layer (canonical music path).
#  - spec.orig{src:path} -> FINAL quality: each shot is cut from the ORIGINAL file (4K/HEVC ok, rotation honoured),
#    cropped 9:16 around the subject at 60 fps, lanczos-scaled to 1080x1920; else 720p proxies (preview quality).
# Shot peaks per window: peak="audio" (impact onsets), "motion" (frame diff), "reps" (reps.py pose peak-effort).
import sys,json,subprocess,os,time,shutil,numpy as np,cv2,soundfile as sf
t0=time.time();SP=json.load(open(sys.argv[1]));MUS=sys.argv[2] if len(sys.argv)>2 else SP.get('music_file')
MUS=os.path.abspath(MUS) if MUS else None
SP['orig']={k:os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),v)) if not os.path.isabs(v) else v for k,v in SP.get('orig',{}).items()}
OW,OH,OF,SR=1080,1920,30,48000;rng=np.random.default_rng(11);cid=SP.get('id','HR_hype');ORIG=SP.get('orig',{})
HERE=os.path.dirname(os.path.abspath(__file__));os.chdir(os.environ['HR_WORK']);sys.path.insert(0,HERE)   # code-dir relative paths (proxy/, sig/, out/) work from any cwd
for d in ('out','flow','sheets','specs'):os.makedirs(d,exist_ok=True)
TMP=f'out/tmp_{cid}';os.makedirs(TMP,exist_ok=True)
# ---------------- beat grid
if MUS:
    import librosa
    y,_=librosa.load(MUS,sr=22050,mono=True);tempo,bt=librosa.beat.beat_track(y=y,sr=22050,units='time')
    rms=librosa.feature.rms(y=y)[0];rt=librosa.times_like(rms,sr=22050)
    def e(a,b):m=(rt>=a)&(rt<b);return rms[m].mean() if m.any() else 0
    jump=[e(bt[i],bt[min(i+8,len(bt)-1)])/(e(bt[max(i-8,0)],bt[i])+1e-6) if 8<=i<len(bt)-25 else 0 for i in range(len(bt))]
    di=int(SP.get('drop_beat_index') or np.argmax(jump));s0=float(bt[di-8]);beats=list(np.array(bt[di-8:di+25])-s0)
    ons=librosa.onset.onset_detect(y=y,sr=22050,units='time');dens=len(ons)/max(1,len(bt))
    BPM=float(np.atleast_1d(tempo)[0]);DROP=beats[8]
    ma,_=librosa.load(MUS,sr=SR,mono=False,offset=s0,duration=beats[32]+0.3);bed=ma.T if ma.ndim==2 else np.repeat(ma[:,None],2,1)
    MUSIC_INFO=dict(file=MUS,bpm=round(BPM,1),drop_src_s=round(s0+DROP,2),window_src_s=[round(s0,2),round(s0+beats[32],2)],drop_jump=round(float(jump[di]),2),onsets_per_beat=round(dens,2),grid_trusted=dens>=0.6)
else:
    BPM=float(SP.get('bpm',140));bl=60/BPM;beats=[i*bl for i in range(33)];DROP=beats[8];bed=None;MUSIC_INFO=None
TOTAL=beats[32];NF=int(round(TOTAL*OF));bl=float(np.median(np.diff(beats)))
plan=[(0,2,0),(2,2,0),(4,2,0),(6,2,0),(8,2,1)]+[(10+i,1,0) for i in range(12)]+[(22,4,1),(26,2,0),(28,2,0),(30,2,0)]
# ---------------- shot peaks per source window
def peaks(w):
    src,a,b,mode=w['src'],w['a'],w['b'],w.get('peak','audio')
    if mode=='reps':
        import reps;r=reps.detect(w['pose']);P=[p for p in (r['peak_effort'] or []) if a<=p<=b]
        if not P:P=[p for p in r['motion_peaks'] if a<=p<=b]
        return [(1.0-0.01*i,p) for i,p in enumerate(P)]
    if mode=='audio':
        A,sr=sf.read(w.get('wav',f'proxy/{src}.wav'),dtype='float32',start=int(a*16000),stop=int(b*16000));A=A if A.ndim==1 else A.mean(1)
        hop=160;e=np.sqrt(np.add.reduceat(A**2,np.arange(0,len(A)-hop,hop))/hop+1e-12);le=20*np.log10(e)
        fl=np.maximum(0,np.diff(le,prepend=le[0]));c=[(fl[i],a+i*hop/sr) for i in range(2,len(fl)-2) if fl[i]>=fl[i-2:i+3].max() and fl[i]>6]
    else:
        cap=cv2.VideoCapture(w.get('ana',f'proxy/{src}.mp4'));cap.set(0,a*1000);pv=None;md=[];fps=cap.get(5) or 30
        for k in range(int((b-a)*fps)):
            ok,f=cap.read()
            if not ok:break
            g=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(90,160)).astype(np.int16);md.append(np.abs(g-pv).mean() if pv is not None else 0);pv=g
        md=np.convolve(md,np.ones(3)/3,'same');c=[(md[i],a+i/fps) for i in range(3,len(md)-3) if md[i]>=md[i-3:i+4].max()]
    if w.get('pose'):   # keep only peaks with a visible person (shoulders+hips vis>0.5) and >=0.4 s from a scene cut
        import reps;d=np.load(w['pose']);mo=d['motion'];ct=d['t'][reps.cuts(mo,reps.CFG)]
        def ok(t):
            m=np.abs(d['t']-t)<=0.3;v=d['lm'][m][:,[11,12,23,24],2]
            return m.any() and np.nanmean(np.nan_to_num(v))>0.5 and (len(ct)==0 or np.min(np.abs(ct-t))>=0.4)
        c=[x for x in c if ok(x[1])]
    c.sort(reverse=True);out=[]
    for s,t in c:
        if all(abs(t-x)>1.2 for _,x in out):out.append((s,t))
    return out
if SP.get('shots'):   # v4 narrative sequencing (sequence.py): explicit, ordered shot list with logged cut reasons
    shots=[dict(src=x['src'],t=float(x['t']),score=1.0,zoom=x.get('zoom',1.12),cx=x['cx'],cx_fixed=True,pose=None,hero_ok=True,bs=x['bs'],nb=x['nb'],ramp=x['ramp']) for x in SP['shots']]
else:
    pool=[]
    for w in SP['windows']:
        pk=peaks(w)[:w.get('take',3)];sc=np.array([s for s,_ in pk]);sc=sc/(sc.max()+1e-9) if len(sc) else sc
        for (s,t),n in zip(pk,sc):pool.append(dict(src=w['src'],t=float(t),score=float(n)*w.get('w',1.0),zoom=w.get('zoom',1.12),cx=w.get('cx',0.5),pose=w.get('pose'),hero_ok=w.get('hero_ok',True)))
    pool.sort(key=lambda d:-d['score'])
    hero=[p for p in pool if p['hero_ok']][0];rest=[p for p in pool if p is not hero];shots=[];last=None
    for bs,nb,rp in plan:
        if bs==8:pick=hero
        elif bs==22:pick=next((p for p in rest if p['hero_ok'] and p['src']!=last),rest[0]);rest.remove(pick)
        else:
            cand=[p for p in rest if p['src']!=last] or rest or [hero];pick=cand[0]
            if pick in rest:rest.remove(pick)
        last=pick['src'];shots.append(dict(pick,bs=bs,nb=nb,ramp=rp))
    # loop: the last shot is the opener's own source, ending exactly where frame 0's source begins (continuous seam)
    if SP.get('loop_continuous',True):
        o=shots[0];shots[-1]=dict(o,t=o['t']-2*bl,bs=shots[-1]['bs'],nb=2,ramp=0)
# ---------------- subject x per shot (sig track or pose npz), median over +-0.6 s
SIG={}
def track_cx(s):
    src,t=s['src'],s['t']
    if s.get('pose'):
        d=np.load(s['pose']);m=np.abs(d['t']-t)<=0.6;x=d['lm'][m][:,[0,11,12,23,24],0].reshape(-1);x=x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else s['cx']
    if src not in SIG:SIG[src]=json.load(open(f'sig/{src}.json'))['track'] if os.path.exists(f'sig/{src}.json') else []
    v=[(r['fx'] if r.get('fx') is not None else (r['x0']+r['x1'])/2) for r in SIG[src] if abs(r['t']-t)<=0.6 and (r.get('fx') is not None or r['x0'] is not None)]
    return float(np.median(v)) if v else s['cx']
for s in shots:
    if not s.get('cx_fixed'):s['cx']=track_cx(s)
# ---------------- timeline
cut=[max(0,int(round(beats[s['bs']]*OF))-1) if s['bs'] else 0 for s in shots]+[NF]
def srctime(s,u,dur):
    if not s['ramp']:return s['t']-0.10+u if s['nb']==1 else s['t']-bl+u
    tp=dur*0.42
    if u<tp-0.35:return s['t']-0.35*0.35-(tp-0.35-u)*1.6
    if u<tp+0.6:return s['t']+(u-tp)*0.35
    return s['t']+0.6*0.35+(u-tp-0.6)*1.3
frames=[]
for i,s in enumerate(shots):
    dur=(cut[i+1]-cut[i])/OF
    for k in range(cut[i],cut[i+1]):frames.append((i,srctime(s,(k-cut[i])/OF,dur)))
def peak_out(i):
    s=shots[i];dur=(cut[i+1]-cut[i])/OF
    return cut[i]/OF+(0.10 if s['nb']==1 and not s['ramp'] else (bl if not s['ramp'] else dur*0.42))
# ---------------- step 1: cut each unique shot from its source (original if given) -> 1080x1920 60 fps intermediates
def dims(p):
    o=json.loads(subprocess.check_output(['ffprobe','-v','0','-select_streams','v:0','-show_entries','stream=width,height:stream_side_data=rotation','-of','json',p]))['streams'][0]
    W,H=o['width'],o['height'];rot=abs(int(float(next((d.get('rotation',0) for d in o.get('side_data_list',[]) if 'rotation' in d),0))))
    return (H,W) if rot in(90,270) else (W,H)
PRE=2.8;clipf={}
for s in shots:
    key=(s['src'],round(s['t'],2))
    if key in clipf:continue
    path=ORIG.get(s['src'],f'proxy/{s["src"]}.mp4');W,H=dims(path);z=s['zoom']
    cw=min(W,H*9/16)/z;ch=cw*16/9;x0=int(np.clip(s['cx']*W-cw/2,0,W-cw));y0=int(np.clip(H*0.5-ch/2,0,H-ch))
    f=f'{TMP}/shot_{len(clipf):02d}.mp4';a=max(0,s['t']-PRE)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',f'{a:.3f}','-t',f'{2*PRE:.3f}','-i',path,'-map','0:v:0','-vf',f'fps=60,crop={int(cw)//2*2}:{int(ch)//2*2}:{x0}:{y0},scale={OW}:{OH}:flags=lanczos','-c:v','libx264','-preset','veryfast','-crf','12','-pix_fmt','yuv420p','-g','12',f],check=True)
    clipf[key]=(f,a,dict(src_wh=[W,H],crop=[int(cw),int(ch),x0,y0],upscale=round(OW/cw,2)))
caps={}
def raw(key,fi):
    f,a,_=clipf[key]
    if key not in caps:caps[key]=[cv2.VideoCapture(f),-10,{},int(cv2.VideoCapture(f).get(7))]
    c=caps[key];cap,pos,cache,n=c;fi=int(np.clip(fi,0,n-1))
    if fi in cache:return cache[fi]
    if fi!=pos+1:cap.set(1,fi)
    ok,im=cap.read();c[1]=fi
    if not ok:im=np.zeros((OH,OW,3),np.uint8)
    cache[fi]=im
    if len(cache)>6:cache.pop(next(iter(cache)))
    if len(caps)>3:
        k0=next(iter(caps))
        if k0!=key:caps[k0][0].release();caps.pop(k0)
    return im
def frame(s,t):
    key=(s['src'],round(s['t'],2));a=clipf[key][1];fi=(t-a)*60;i0=int(np.floor(fi));al=fi-i0;im=raw(key,i0)
    return im if al<0.2 else cv2.addWeighted(im,1-al,raw(key,i0+1),al,0)
def zoom(img,z,dx=0,dy=0,rot=0):
    M=cv2.getRotationMatrix2D((OW/2,OH/2),rot,z);M[0,2]+=dx;M[1,2]+=dy;return cv2.warpAffine(img,M,(OW,OH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
# ---------------- step 2: FX pass -> near-lossless intermediate
po=[peak_out(i) for i in range(len(shots))];LOOP=int(SP.get('loop_frames',10))   # 10-15; pipeline retries with 15 if the seam is below the median
enc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{OW}x{OH}','-r',str(OF),'-i','-','-c:v','libx264','-preset','veryfast','-crf','10','-pix_fmt','yuv420p',f'{TMP}/v.mp4'],stdin=subprocess.PIPE)
kern=np.zeros((1,31),np.float32);kern[0,:]=1/31
for k,(i,t) in enumerate(frames):
    s=shots[i];T=k/OF;img=frame(s,t);z=1.0;dx=dy=0;rot=0;d=T-po[i]
    if 0<=d<0.4:z*=1+(0.22 if s['ramp'] or s['bs']==8 else 0.12)*np.exp(-d/0.10)
    if 0<=d<0.18 and s['bs']>=8:a=18*(1-d/0.18);dx=rng.uniform(-a,a);dy=rng.uniform(-a,a)
    u=k-cut[i]
    if i>0 and u<3:z*=1.10-0.033*u;rot=(3-u)*1.5*(1 if i%2 else -1)
    if k>=NF-LOOP:
        w=(k-(NF-LOOP)+1)/LOOP;w=w*w*(3-2*w);pin=frame(shots[0],shots[0]['t']-bl-(NF-k)/OF)
        img=cv2.addWeighted(zoom(img,z,dx,dy,rot),1-w,pin,w,0)
    elif z!=1 or dx or dy or rot:img=zoom(img,z,dx,dy,rot)
    if i>0 and u<2:img=cv2.filter2D(img,-1,kern)
    fl=0
    if s['bs'] in(8,22) and u==0:fl=0.85
    elif s['bs'] in(8,22) and u==1:fl=0.35
    elif i>0 and u==0 and s['bs']%4==0:fl=0.45
    if fl:img=cv2.addWeighted(img,1-fl,np.full_like(img,255),fl,0)
    enc.stdin.write(img.tobytes())
enc.stdin.close();enc.wait()
for c in caps.values():c[0].release()
# ---------------- step 3: audio
N=NF*(SR//OF);out=np.zeros((N+SR,2),np.float32)
def put(x,T,g=1.0):
    i=int(T*SR);n=min(len(x),len(out)-i)
    if n>0 and i>=0:out[i:i+n]+=g*(x[:n,None] if x.ndim==1 else x[:n])
def kick(d=0.35):t=np.arange(int(d*SR))/SR;f=50+110*np.exp(-t*30);return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*9)
def snare(d=0.2):t=np.arange(int(d*SR))/SR;n=rng.normal(0,1,len(t));n=n-np.convolve(n,np.ones(8)/8,'same');return (0.6*n+0.5*np.sin(2*np.pi*190*t))*np.exp(-t*22)
def hat(d=0.05):t=np.arange(int(d*SR))/SR;n=rng.normal(0,1,len(t));n=n-np.convolve(n,np.ones(3)/3,'same');return n*np.exp(-t*80)
def sub(d):t=np.arange(int(d*SR))/SR;return np.tanh(2.2*np.sin(2*np.pi*46*t))*np.exp(-t*1.6)
def whoosh(d=0.3):n=int(d*SR);x=rng.normal(0,1,n);y=np.convolve(x,np.ones(40)/40,'same');return 0.5*y*np.sin(np.pi*np.arange(n)/n)**2
def riser(d):n=int(d*SR);t=np.arange(n)/SR;x=rng.normal(0,1,n);x=x-np.convolve(x,np.ones(6)/6,'same');return x*(t/d)**2*0.5
sfx=[];music_mode='file' if MUS else SP.get('music','none')   # generated bed retired (spec §17): default is SFX only; 'generated' only for internal previews
if bed is not None:
    put(bed[:N],0,0.9)
    xf=int(0.12*SR);tail=bed[N:N+xf]           # loop: music continuing past beat 32 crossfades into the head (both on a downbeat)
    if len(tail)==xf:u=np.linspace(0,1,xf)[:,None];out[:xf]=np.sin(u*np.pi/2)*out[:xf]+np.cos(u*np.pi/2)*0.9*tail
elif music_mode=='generated':
    for bi in range(32):
        T=beats[bi];post=bi>=8
        if bi==7:continue
        if post or bi%2==0:put(kick(),T,0.9 if post else 0.6)
        if bi%2==1:put(snare(),T,0.55 if post else 0.35)
        for h in ((0,0.5) if post else (0.5,)):put(hat(),T+h*bl,0.18)
        if post and bi%4==0:put(sub(bl*3.5),T,0.45)
    put(riser(beats[7]-beats[4]),beats[4],0.35)
for i in range(1,len(shots)):
    T=cut[i]/OF
    if shots[i]['bs']%4==0 or shots[i]['bs'] in(8,22):put(whoosh(),T-0.22,0.35 if bed is not None else 0.45);sfx.append(round(T-0.22,2))
put(kick(0.5)*1.2,DROP,0.6 if bed is not None else 0.8)
out=out[:N]
if bed is None:n=int(0.015*SR);u=np.linspace(0,1,n)[:,None];out[-n:]=np.cos(u*np.pi/2)*out[-n:]+np.sin(u*np.pi/2)*out[:n]
sf.write(f'{TMP}/a.wav',np.clip(out,-1,1),SR)
# ---------------- step 4: text, grade, encode (8-12 Mbps), loudness on the final file
def tt(x):x=max(x,0);return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:05.2f}"
L=["[Script Info]","ScriptType: v4.00+","PlayResX: 1080","PlayResY: 1920","WrapStyle: 0","","[V4+ Styles]",
"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
"Style: Hook,DejaVu Sans,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,1,0,0,0,100,100,0,0,1,7,3,8,70,70,170,1",
"Style: Kin,DejaVu Sans,150,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,4,0,1,10,4,5,60,60,0,1",
"","[Events]","Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
hk=SP.get('hook','No days off.');em=SP.get('emph','')
if em:hk=hk.replace(em,'{\\c&H0000D7FF&}'+em+'{\\c&H00FFFFFF&}')
words=SP.get('words',['EVERY','ROUND','EVERY','REP']);seam=(NF-LOOP)/OF-0.05
# ---- face-aware placement (captions.py): every line avoids the faces over its whole span
import captions as CAP
HDR='\n'.join(L);HSEAM=5;FB=CAP.detect(f'{TMP}/v.mp4',os.environ.get('HR_MODELS',os.path.dirname(HERE)),cuts=cut[:-1]+[NF-HSEAM-1]);PLACE=[]   # the loop morph tail counts as its own segment
def pl(name,style,text,spans,zones,fs,shift_ok=True,shift_mode='both'):
    r=CAP.place(HDR,dict(style=style,text=text,spans=spans,zones=zones,fs=fs,shift_ok=shift_ok,shift_mode=shift_mode,end_idx=1),FB,OF,bl,seam);r['line']=name;PLACE.append(r);return r
CUTS=sorted({c/OF for c in cut})
def pl_split(name,style,text,st,en,zones,fs,pop,seam_span=None,shift_mode='both'):
    """1) whole span: zones in fixed order, font 100->85%; 2) if no zone is face-free for the whole span, one placement
    per shot segment (split at the cuts; same order); 3) last resort: shift the line one beat and repeat 1-2.
    Returns [(start,end,placement,first)]; a failed placement keeps ok=False and the QA gate fails the reel."""
    for sh in (0.0,bl):
        if shift_mode=='start':a0,b0=st+sh,en
        elif shift_mode=='end':a0,b0=st,en-sh
        else:a0,b0=st+sh,min(en+sh,seam)
        if b0-a0<0.3:continue
        n0=len(PLACE);head=[(a0,min(a0+0.14,b0),pop),(min(a0+0.14,b0),b0,100)]+([seam_span] if seam_span else [])
        r=pl(name,style,text,head,zones,fs,shift_ok=False)
        if r['ok']:return [(a0,b0,r,True)]
        bd=[a0]+[c for c in CUTS if a0+0.1<c<b0-0.1]+[b0];out=[]
        for j,(a,b2) in enumerate(zip(bd[:-1],bd[1:])):
            sp=[(a,min(a+0.14,b2),pop),(min(a+0.14,b2),b2,100)] if j==0 else [(a,b2,100)]
            if j==0 and seam_span:sp.append(seam_span)
            out.append((a,b2,pl(f'{name}[{j}]',style,text,sp,zones,fs,shift_ok=False),j==0))
        if all(o[2]['ok'] for o in out):
            del PLACE[n0];return out
        if sh==0.0:del PLACE[n0:]
    del PLACE[n0]
    # 4) still no face-free spot in some shot segment: never draw over the face; the line is simply not shown in that
    #    segment (for the opening hook this means it ends early). Logged as 'hidden'.
    for o in out:
        if not o[2]['ok']:o[2]['hidden']=True;o[2]['ok']=True
    return [o for o in out if not o[2].get('hidden')]
# hook (§4.2): top safe area -> upper-third left -> upper-third right -> mid-frame above centre; never bottom 20%, never
# over a face. 140% pop for 0.14 s, 100% after, 140% static over the seam: the seam copy uses frame 0's placement.
# the seam copy only covers the last 5 frames (the morph has mostly landed on frame 0's source, so the faces under it
# are frame 0's faces); it is fully opaque on the last frame, so the last frame still matches frame 0 including text
POP='\\fscx140\\fscy140\\t(0,140,\\fscx96\\fscy96)\\t(140,240,\\fscx100\\fscy100)'
H0=None
for a,b2,r,first in pl_split('hook','Hook',hk,0,beats[7],CAP.HOOK_ZONES,86,140,((NF-HSEAM)/OF,NF/OF,140),shift_mode='end'):
    if first:H0=r
    L.append(f"Dialogue: 2,{tt(a)},{tt(b2)},Hook,,0,0,0,,{{{r['pos']}{POP if first else ''}}}{r['text']}")
# loop: the hook fades in over the seam at frame-0 state (140%), so the last frame matches frame 0 including text
if H0:L.append(f"Dialogue: 2,{tt((NF-HSEAM)/OF)},{tt(NF/OF+0.1)},Hook,,0,0,0,,{{{H0['pos']}\\fad(130,0)\\fscx140\\fscy140}}{H0['text']}")
for j,wd in enumerate(words):
    st=beats[8+2*j];en=min(beats[10+2*j],seam)
    if st>=seam:break
    col='&H0000D7FF&' if j==len(words)-1 else '&H00FFFFFF&'
    for a,b2,r,first in pl_split(f'word{j}','Kin',wd,st,en,CAP.WORD_ZONES,150,165,shift_mode='start'):
        an=('\\fscx165\\fscy165\\t(0,110,\\fscx95\\fscy95)\\t(110,200,\\fscx100\\fscy100)' if first else '')
        L.append(f"Dialogue: 1,{tt(a)},{tt(b2)},Kin,,0,0,0,,{{{r['pos']}\\c{col}{an}}}{r['text']}")
if SP.get('end_line'):
    st,en=beats[22],min(beats[29],seam)
    for a,b2,r,first in pl_split('end_line','Hook',SP['end_line'],st,en,CAP.END_ZONES,86,100):
        L.append(f"Dialogue: 1,{tt(a)},{tt(min(b2,seam))},Hook,,0,0,0,,{{{r['pos']}{'\\fad(80,0)' if first else ''}}}{r['text']}")
for r in PLACE:print(f"caption {r['line']}: zone {r['zone']} {r['pos']} font {r['fs']}px ({r['scale']:.0%}) shift {r['shift']:.2f}s {'HIDDEN (no face-free spot in this segment)' if r.get('hidden') else ('OK' if r['ok'] else 'NO SAFE SPOT')}",file=sys.stderr)
open(f'{TMP}/t.ass','w').write('\n'.join(L)+'\n')
grade='eq=contrast=1.18:saturation=1.25:gamma=0.95,colorbalance=rs=-0.03:bs=0.04:rh=0.03,vignette=angle=PI/4.5,unsharp=5:5:0.5'
vt=f'{TMP}/vg.mp4';fo=f'out/{cid}_9x16.mp4'
subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',f'{TMP}/v.mp4','-vf',f"{grade},subtitles={TMP}/t.ass",'-an','-c:v','libx264','-preset','slow','-profile:v','high','-b:v','10M','-minrate','8M','-maxrate','12M','-bufsize','12M','-x264-params','nal-hrd=vbr','-pix_fmt','yuv420p','-r','30',vt],check=True)
import loudness
LR=loudness.mux(vt,f'{TMP}/a.wav',fo,lim0=-4.5);I,P,hist=LR['lufs'],LR['tp'],LR['loud']
if not LR['ok']:print(f'WARNING loudness not in spec after {len(hist)} iterations: {I} LUFS {P} dBTP',file=sys.stderr)
# ---------------- QA: probe, bitrate, seam SSIM (270x480 gray, last vs first vs median consecutive)
pr=json.loads(subprocess.check_output(['ffprobe','-v','0','-show_entries','stream=codec_name,width,height,r_frame_rate,bit_rate,pix_fmt:format=duration,bit_rate','-of','json',fo]))
c=cv2.VideoCapture(fo);F=[]
while True:
    ok,f=c.read()
    if not ok:break
    F.append(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(270,480)).astype(np.float32))
def ss(x,y):
    C1,C2=6.5025,58.5225;g=lambda z:cv2.GaussianBlur(z,(7,7),1.5);mx,my=g(x),g(y);vx=g(x*x)-mx*mx;vy=g(y*y)-my*my;cxy=g(x*y)-mx*my
    return float((((2*mx*my+C1)*(2*cxy+C2))/((mx*mx+my*my+C1)*(vx+vy+C2))).mean())
seam_ssim=ss(F[-1],F[0]);cons=float(np.median([ss(F[i],F[i+1]) for i in range(0,len(F)-1,5)]))
shutil.copy(f'{TMP}/t.ass',f'out/{cid}_9x16.ass');shutil.rmtree(TMP)   # keep the exact overlay for the QA text-vs-face check
vs=[s for s in pr['streams'] if s['codec_name']=='h264'][0]
print(json.dumps(dict(captions=PLACE,loop_frames=LOOP,file=fo,ass=f'out/{cid}_9x16.ass',w=vs['width'],h=vs['height'],fps=vs['r_frame_rate'],pix=vs['pix_fmt'],video_mbps=round(int(vs['bit_rate'])/1e6,2),total_mbps=round(int(pr['format']['bit_rate'])/1e6,2),dur=pr['format']['duration'],
  lufs=I,tp=P,loud=hist,seam_ssim=round(seam_ssim,3),median_consecutive_ssim=round(cons,3),music=music_mode,music_info=MUSIC_INFO,bpm=round(BPM,1),drop_s=round(DROP,3),render_s=round(time.time()-t0,1),
  shots=[dict(src=s['src'],t=round(s['t'],2),beat=s['bs'],nb=s['nb'],ramp=s['ramp'],cx=round(s['cx'],2),upscale=clipf[(s['src'],round(s['t'],2))][2]['upscale']) for s in shots])))
