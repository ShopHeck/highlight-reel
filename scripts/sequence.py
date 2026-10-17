# Narrative sequencing layer for the hype montage (v4). usage: python sequence.py specs/HR_hype_final.json out_spec.json
# 1) candidate moments (audio/motion peaks per window)  2) tag phase/direction/scale/position/energy from pose+flow
# 3) beam-search an order on the music energy arc with continuity rules  4) log a reason per cut, score flow vs the old render
import sys,json,numpy as np,soundfile as sf,cv2
import seq_features as F
import os
SP=json.load(open(sys.argv[1]));OUT=os.path.abspath(sys.argv[2])
HERE=os.path.dirname(os.path.abspath(__file__));os.chdir(os.environ['HR_WORK'])   # work dir holds proxy/, flow/, out/
for d in ('out','flow','sheets','specs',os.path.dirname(OUT)):os.makedirs(d,exist_ok=True)
BPM=float(SP.get('bpm',140));bl=60/BPM;OF=30
beats=[i*bl for i in range(33)]
PLAN=[(0,2,0),(2,2,0),(4,2,0),(6,2,0),(8,2,1)]+[(10+i,1,0) for i in range(12)]+[(22,4,1),(26,2,0),(28,2,0),(30,2,0)]
# music sections -> target energy (0..1), allowed phases, target shot scale (0 wide,1 medium,2 close, None=free)
def slot(bs):
    if bs==0: return dict(sec='intro',E=.85,ph=['peak','grind'],sc=None)   # v5: frame 0 = strong moment (loop shot is its own lead-in)
    if bs==2: return dict(sec='intro',E=.3,ph=['warm-up','setup'],sc=0)
    if bs==4: return dict(sec='riser',E=.5,ph=['build','warm-up'],sc=1)
    if bs==6: return dict(sec='riser',E=.65,ph=['build'],sc=2)
    if bs==8: return dict(sec='drop',E=1.0,ph=['peak'],sc=None)
    if bs<14: return dict(sec='drive',E=.72,ph=['grind','build','peak'],sc=None)
    if bs<18: return dict(sec='drive',E=.82,ph=['grind','peak'],sc=None)
    if bs<22: return dict(sec='climax',E=.92,ph=['peak','grind'],sc=None)
    if bs==22:return dict(sec='hero',E=1.0,ph=['payoff','peak'],sc=None)
    if bs==26:return dict(sec='resolve',E=.6,ph=['payoff','grind'],sc=1)
    if bs==28:return dict(sec='resolve',E=.4,ph=['payoff','warm-up','build'],sc=0)
    return dict(sec='loop',E=None,ph=None,sc=None)
# v5 session / wardrobe continuity. Sessions = shooting session + outfit (same gym, same clothes).
SESSION={'a':'jacket-bag','b':'jacket-bag','copyF25A':'jacket-bag','IMG_0089':'shirtless-bag','IMG_0094':'shirtless-bag'}
sess=lambda src:SESSION.get(src,'pads' if src.startswith('video-') else src)
def group(bs):   # music blocks that must stay in ONE session: intro | riser | pad round (drop..hero) | resolve+loop
    return 'intro' if bs<4 else 'riser' if bs<8 else 'round' if bs<26 else 'resolve'
def sesscost(seq,s):
    if not seq:return 0.0
    A=seq[-1];same=sess(A['src'])==sess(s['src'])
    if group(A['bs'])==group(s['bs']):return 0.0 if same else 4.0
    return 0.0 if same else 0.8
def srctime(s,u,dur):
    if not s['ramp']:return s['t']-0.10+u if s['nb']==1 else s['t']-bl+u
    tp=dur*0.42
    if u<tp-0.35:return s['t']-0.35*0.35-(tp-0.35-u)*1.6
    if u<tp+0.6:return s['t']+(u-tp)*0.35
    return s['t']+0.6*0.35+(u-tp-0.6)*1.3
def durof(bs,nb):a=max(0,int(round(beats[bs]*OF))-1) if bs else 0;b=max(0,int(round(beats[bs+nb]*OF))-1) if bs+nb<32 else int(round(beats[32]*OF));return (b-a)/OF
# ---------- candidates
def peaks(w):
    src,a,b,mode=w['src'],w['a'],w['b'],w.get('peak','audio')
    if mode=='audio':
        A,sr=sf.read(f'proxy/{src}.wav',dtype='float32',start=int(a*16000),stop=int(b*16000));A=A if A.ndim==1 else A.mean(1)
        hop=160;e=np.sqrt(np.add.reduceat(A**2,np.arange(0,len(A)-hop,hop))/hop+1e-12);le=20*np.log10(e)
        fl=np.maximum(0,np.diff(le,prepend=le[0]));c=[(fl[i],a+i*hop/sr) for i in range(2,len(fl)-2) if fl[i]>=fl[i-2:i+3].max() and fl[i]>6]
    else:
        cap=cv2.VideoCapture(f'proxy/{src}.mp4');cap.set(0,a*1000);pv=None;md=[];fps=cap.get(5) or 30
        for k in range(int((b-a)*fps)):
            ok,f=cap.read()
            if not ok:break
            g=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(90,160)).astype(np.int16);md.append(np.abs(g-pv).mean() if pv is not None else 0);pv=g
        md=np.convolve(md,np.ones(3)/3,'same');c=[(md[i],a+i/fps) for i in range(3,len(md)-3) if md[i]>=md[i-3:i+4].max()]
    c.sort(reverse=True);out=[]
    for s,t in c:
        if all(abs(t-x)>1.2 for _,x in out):out.append((s,t))
    return out
def geom(tr,zoom,cx):
    W,H=tr['W'],tr['H'];cw=min(W,H*9/16)/zoom/W;x0=float(np.clip(cx-cw/2,0,1-cw));ch=cw*W*16/9/H;y0=(1-ch)/2;return cw,x0,ch,y0
def state(c,t0,t1):
    """subject + motion state of a shot over source interval [t0,t1] as seen ON SCREEN (after the 9:16 crop)"""
    tr=c['tr'];cw,x0,ch,y0=geom(tr,c['zoom'],c['cx'])
    sx=F.at(tr,t0,t1,'cx');hh=[(r['y1']-r['y0'])/ch for r in tr['rec'] if r and 'y0' in r and t0<=r['t']<=t1]
    ank=[r['ank'] for r in tr['rec'] if r and 'ank' in r and t0<=r['t']<=t1];hip=[r['hip'] for r in tr['rec'] if r and 'hip' in r and t0<=r['t']<=t1]
    ws,wdx=F.wrist_speed(tr,t0,t1);fm=F.at(tr,t0,t1,'mag') or 0;fdx=F.at(tr,t0,t1,'dx') or 0
    h=float(np.median(hh)) if hh else None
    sc=None if h is None else (0 if (np.mean(ank)>0.5 and h<0.8) else (1 if np.mean(hip)>0.5 and h<1.05 else 2))
    d=(fdx/(abs(fdx)+0.15))*0.5+np.tanh(wdx/0.8)*0.5    # -1 left .. +1 right (screen direction of the action)
    return dict(x=None if sx is None else float((sx-x0)/cw),h=h,sc=sc,mot=float(ws+0.6*fm),dir=float(d))
def inframe(c,t0,t1):
    r=[x for x in c['tr']['rec'] if x and t0<=x['t']<=t1];
    if not r:return 0
    cw,x0,_,_=geom(c['tr'],c['zoom'],c['cx']);return float(np.mean([('cx' in x) and (x0-0.02<=x['cx']<=x0+cw+0.02) for x in r]))
def build_pool(windows,take):
    P=[]
    for w in windows:
        for s,t in peaks(w)[:take]:
            tr=F.track(w['src'],float(t));c=dict(src=w['src'],t=round(float(t),2),zoom=w.get('zoom',1.15),w=w.get('w',1),tr=tr)
            v=[r['cx'] for r in tr['rec'] if r and 'cx' in r and abs(r['t']-t)<=0.6];c['cx']=float(np.median(v)) if v else w.get('cx',0.5)
            up0=None
            for f in SP.get('scale_variants',[1.0,1.3,1.6]):   # tighter crops of the SAME original = shot-scale options
                z=c['zoom']*f;up=upscale(w['src'],z)
                if up>SP.get('max_upscale',1.5):continue
                v=dict(c,zoom=round(z,3),up=round(up,2))
                h=state(v,t-0.3,t+0.3)['h']
                if f>1 and (h is None or h>1.3):continue
                P.append(v)
    F.save();return P
import subprocess
_D={}
def upscale(src,z):
    if src not in _D:
        p=SP['orig'].get(src,f'proxy/{src}.mp4');o=json.loads(subprocess.check_output(['ffprobe','-v','0','-select_streams','v:0','-show_entries','stream=width,height:stream_side_data=rotation','-of','json',p]))['streams'][0]
        W,H=o['width'],o['height'];rot=abs(int(float(next((d.get('rotation',0) for d in o.get('side_data_list',[]) if 'rotation' in d),0))));_D[src]=(H,W) if rot in(90,270) else (W,H)
    W,H=_D[src];return 1080/(min(W,H*9/16)/z)
def tag(P):
    def m(c,a,b):return state(c,c['t']+a,c['t']+b)['mot']
    for c in P:
        c['E_raw']=m(c,-0.4,0.4);c['pre']=m(c,-1.5,-0.5);c['post']=m(c,0.5,1.5);c['sus']=m(c,-1.5,2.0)
        st=state(c,c['t']-0.4,c['t']+0.4);c.update(sc=st['sc'],x=st['x'],dir=st['dir'])
    E=np.array([c['E_raw'] for c in P]);r=E.argsort().argsort()/(len(E)-1)
    srcE={s:np.median([c['E_raw'] for c in P if c['src']==s]) for s in {c['src'] for c in P}}
    for c,p in zip(P,r):
        c['E']=float(p);burst=c['E_raw']/(c['sus']+1e-6);rel=(c['post']-c['pre'])/(c['sus']+1e-6)
        if p<0.18:ph='setup'
        elif p<0.4:ph='warm-up'
        elif p>=0.8 and burst>=1.15:ph='peak'
        elif p>=0.55 and rel<=-0.25:ph='payoff'
        elif rel>=0.2:ph='build'
        elif p>=0.55:ph='grind'
        else:ph='build'
        c['phase']=ph;c['burst']=round(float(burst),2);c['trend']=round(float(rel),2)
    return P
def corr(a,b):
    if a is None or b is None:return 0
    a=np.array(a)-np.mean(a);b=np.array(b)-np.mean(b);return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def shot(c,bs,nb,ramp):return dict(c,bs=bs,nb=nb,ramp=ramp)
def io(s):
    d=durof(s['bs'],s['nb']);a=srctime(s,0,d);b=srctime(s,d,d);return a,b
def trans(A,B,sec):
    """continuity of cut A->B: returns (score 0..1, components, reason)"""
    a0,a1=io(A);b0,b1=io(B);sa=state(A,a1-0.3,a1);sb=state(B,b0,b0+0.3)
    pos=1.0 if sa['x'] is None or sb['x'] is None else max(0.0,1-abs(sa['x']-sb['x'])/0.35)
    both=abs(sa['dir'])>0.3 and abs(sb['dir'])>0.3;dirs=0.0 if both and np.sign(sa['dir'])!=np.sign(sb['dir']) else 1.0
    act=float(np.clip((min(sa['mot'],sb['mot'])-2.3)/2.2,0,1))   # calibrated on the pool: p10 2.3, p75 5.1
    st=None if sa['sc'] is None or sb['sc'] is None else sb['sc']-sa['sc']
    if st is None:scl=0.5
    elif sec in('intro','riser','drop'):scl=1.0 if st in(0,1) else (0.5 if st==2 else 0.0)
    elif sec in('resolve','loop'):scl=1.0 if st in(0,-1) else 0.4
    else:scl=1.0 if abs(st)<=1 else 0.5
    dup=0.0 if (A['src']==B['src'] and abs(A['t']-B['t'])<3) or corr(A['tr']['thumb'],B['tr']['thumb'])>0.93 else 1.0
    S=0.3*pos+0.25*dirs+0.25*act+0.1*scl+0.1*dup
    nm={0:'wide',1:'medium',2:'close',None:'?'}
    why=[]
    if act>=0.7:why.append('match-cut on '+('punch extension' if sb['mot']>=5.0 else 'continuing movement')+f" (motion {sa['mot']:.1f}→{sb['mot']:.1f} across the cut)")
    if dirs==1 and both:why.append(f"screen direction held ({'→' if sa['dir']>0 else '←'})")
    if st is not None and st>0:why.append(f"scale step {nm[sa['sc']]}→{nm[sb['sc']]}"+(' into drop' if B['bs']==8 else ''))
    elif st is not None and st<0 and sec in('resolve','loop'):why.append(f"scale opens {nm[sa['sc']]}→{nm[sb['sc']]} to resolve")
    if sess(A['src'])==sess(B['src']):why.append(f"same session ({sess(B['src'])})")
    else:why.append(f"session change {sess(A['src'])}→{sess(B['src'])} at section boundary")
    if pos>=0.8 and sa['x'] is not None and sb['x'] is not None:why.append(f"subject zone held (x {sa['x']:.2f}→{sb['x']:.2f})")
    return S,dict(pos=round(pos,2),dir=dirs,act=round(act,2),scale=scl,dup=dup,xa=sa['x'],xb=sb['x'],da=round(sa['dir'],2),db=round(sb['dir'],2),ma=round(sa['mot'],2),mb=round(sb['mot'],2),sca=sa['sc'],scb=sb['sc']),why
def flow_report(shots):
    T=[];arc=[]
    for i in range(1,len(shots)):S,c,_=trans(shots[i-1],shots[i],slot(shots[i]['bs'])['sec']);T.append((S,c))
    for s in shots:
        if slot(s['bs'])['E'] is not None:arc.append((slot(s['bs'])['E'],s.get('E',0)))
    a=np.array(arc);sp=float(np.corrcoef(a[:,0].argsort().argsort(),a[:,1].argsort().argsort())[0,1])
    k=lambda n:round(float(np.mean([c[n] for _,c in T])),3)
    return dict(flow=round(100*float(np.mean([S for S,_ in T])),1),pos=k('pos'),dir=k('dir'),act=k('act'),scale=k('scale'),dup=k('dup'),arc_rank_corr=round(sp,2),
                phase_fit=round(float(np.mean([s.get('phase') in slot(s['bs'])['ph'] for s in shots[:-1]])),2),
                session_jumps_in_block=int(sum(group(a['bs'])==group(b['bs']) and sess(a['src'])!=sess(b['src']) for a,b in zip(shots[:-1],shots[1:]))),
                session_changes=int(sum(sess(a['src'])!=sess(b['src']) for a,b in zip(shots[:-1],shots[1:]))),frame0_E=round(float(shots[0].get('E',0)),2))
if __name__=='__main__':
    WIN=SP['windows']+SP.get('extra_windows',[])
    P=tag(build_pool(WIN,SP.get('seq_take',8)))
    P=[c for c in P if inframe(c,c['t']-0.6,c['t']+0.6)>=0.8]     # subject-in-frame rule (>=0.8)
    if SP.get('compare_spec') and SP.get('exclude_compare_moments'):   # optional: new shot selection, no moment of the compare render reused
        EX=json.load(open(SP['compare_spec']))['shots'];P=[c for c in P if not any(c['src']==e['src'] and abs(c['t']-e['t'])<1.5 for e in EX)]
    print('pool',len(P),{p:sum(c['phase']==p for c in P) for p in ['setup','warm-up','build','grind','peak','payoff']})
    # ---- beam search
    def slotcost(c,bs,nb,ramp):
        sl=slot(bs);s=shot(c,bs,nb,ramp);a,b=io(s)
        if inframe(c,a,b)<0.8:return None
        k=2.0*abs(c['E']-sl['E'])+(0 if c['phase'] in sl['ph'] else (0.6 if bs not in (0,8) else 3))
        if sl['sc'] is not None and c['sc'] is not None:k+=0.25*abs(c['sc']-sl['sc'])
        return k
    beam=[(0.0,[],set())]
    for bs,nb,ramp in PLAN[:-1]:
        nxt=[]
        for cost,seq,used in beam:
            for i,c in enumerate(P):
                if i in used or (seq and seq[-1]['src']==c['src']):continue
                if bs==28 and c['src']==seq[0]['src']:continue
                if bs>=26 and sess(c['src'])!=sess(seq[0]['src']):continue   # resolve block ends in the loop shot = opener's session   # shot 28 neighbours the loop shot (= opener's source)
                if any(s['src']==c['src'] and abs(s['t']-c['t'])<2.5 for s in seq):continue
                k=slotcost(c,bs,nb,ramp)
                if k is None:continue
                s=shot(c,bs,nb,ramp);tc=sesscost(seq,s)
                if tc>=4.0:continue   # hard: one session per music block (a whole block may use any session)
                if seq:S,_,_=trans(seq[-1],s,slot(bs)['sec']);tc=2.5*(1-S)
                if bs==28:   # close the loop: shot 28 -> the opener's own source
                    o=dict(seq[0]);L=shot(dict(o,t=round(o['t']-2*bl,2)),30,2,0);S,_,_=trans(s,L,'loop');tc+=2.5*(1-S)+sesscost(seq+[s],L)
                nxt.append((cost+k+tc,seq+[s],used|{i}))
        nxt.sort(key=lambda x:x[0]);beam=nxt[:int(SP.get('beam',120))]
        print(bs,round(beam[0][0],2),beam[0][1][-1]['src'],beam[0][1][-1]['t'],flush=True)
    seq=beam[0][1]
    # ---- loop seam: with no previous reel to compare against, score candidate openers/loop shots by the ACTUAL
    # last-to-first SSIM (the last frame is the opener's source frame 1/30 s before frame 0) and keep the best one.
    HAVE_PREV=bool(SP.get('compare_spec') and os.path.exists(SP['compare_spec'])) or os.path.exists('out/previous.qa.json')
    def _g(src,t,cx,z):
        cap=cv2.VideoCapture(f'proxy/{src}.mp4');fps=cap.get(5) or 30;W=cap.get(3);H=cap.get(4);cap.set(1,max(0,int(round(t*fps))));ok,f=cap.read();cap.release()
        if not ok:return None
        cw=min(W,H*9/16)/z;x0=int(np.clip(cx*W-cw/2,0,W-cw));y0=int(max(0,H/2-cw*16/18));f=f[y0:y0+int(cw*16/9),x0:x0+int(cw)]
        return cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(270,480)).astype(np.float32)
    def _ss(x,y):
        C1,C2=6.5025,58.5225;g=lambda z:cv2.GaussianBlur(z,(7,7),1.5);mx,my=g(x),g(y);vx=g(x*x)-mx*mx;vy=g(y*y)-my*my;cxy=g(x*y)-mx*my
        return float((((2*mx*my+C1)*(2*cxy+C2))/((mx*mx+my*my+C1)*(vx+vy+C2))).mean())
    def seam_ssim(c):
        t0=c['t']-bl;a=_g(c['src'],t0-1/OF,c['cx'],c['zoom']);b=_g(c['src'],t0,c['cx'],c['zoom'])
        return None if a is None or b is None else _ss(a,b)
    def cons(c):
        v=[]
        for u in (0.0,0.2,0.4):
            a=_g(c['src'],c['t']+u,c['cx'],c['zoom']);b=_g(c['src'],c['t']+u+1/OF,c['cx'],c['zoom'])
            if a is not None and b is not None:v.append(_ss(a,b))
        return float(np.median(v)) if v else 0.6
    if not HAVE_PREV:
        med=float(np.median([cons(x) for x in seq]));cur=seq[0];last=seq[-1];res=[]
        S1,_,_=trans(cur,seq[1],slot(2)['sec']);L0=shot(dict(cur,t=round(cur['t']-2*bl,2)),30,2,0);S2,_,_=trans(last,L0,'loop')
        for c in [cur]+[c for c in P if c['phase'] in slot(0)['ph'] and c['src'] not in (seq[1]['src'],last['src']) and sess(c['src'])==sess(seq[1]['src'])
                     and not any(x['src']==c['src'] and abs(x['t']-c['t'])<2.5 for x in seq[1:])]:
            o=shot(c,0,2,0)
            if inframe(c,*io(o))<0.8:continue
            L=shot(dict(c,t=round(c['t']-2*bl,2)),30,2,0)
            if inframe(L,*io(L))<0.8:continue
            sv=seam_ssim(c)
            if sv is None:continue
            a,_,_=trans(o,seq[1],slot(2)['sec']);b,_,_=trans(last,L,'loop')
            res.append((sv-0.5*max(0,S1-a)-0.5*max(0,S2-b)-0.3*abs(c['E']-cur['E']),sv,c))
        res.sort(key=lambda r:-r[0])
        print('loop-seam candidates (score, seam ssim, src, t):',[(round(r[0],3),round(r[1],3),r[2]['src'],r[2]['t']) for r in res[:6]],'est median',round(med,3),flush=True)
        if res and res[0][2] is not cur:seq[0]=shot(res[0][2],0,2,0)
        SEAM=dict(est_seam_ssim=round(res[0][1],3) if res else None,est_median=round(med,3),candidates=len(res))
    else:SEAM=None
    o=seq[0];oc=dict(o,t=round(o['t']-2*bl,2));oc['tr']=F.track(oc['src'],oc['t']);seq.append(shot(oc,30,2,0));F.save()
    # ---- old render, scored with the same features
    # previous render to compare against: spec.compare_spec (a sequenced spec) > out/previous.qa.json > none
    OLD=[];zm={w['src']:w.get('zoom',1.12) for w in SP['windows']}
    if SP.get('compare_spec') and os.path.exists(SP['compare_spec']):
        for s in json.load(open(SP['compare_spec']))['shots']:
            c=dict(src=s['src'],t=s['t'],zoom=s.get('zoom',zm.get(s['src'],1.12)),cx=s['cx'],tr=F.track(s['src'],s['t']));OLD.append(shot(c,s.get('bs',s.get('beat')),s['nb'],s['ramp']))
    elif os.path.exists('out/previous.qa.json'):
        for s in json.load(open('out/previous.qa.json'))['shots']:
            c=dict(src=s['src'],t=s['t'],zoom=zm.get(s['src'],1.12),cx=s['cx'],tr=F.track(s['src'],s['t']));OLD.append(shot(c,s['beat'],s['nb'],s['ramp']))
    tag(OLD+P)  # same energy normalisation
    F.save()
    tag(P+OLD);rep_old=flow_report(OLD) if OLD else None;rep_new=flow_report(seq)
    cuts=[];
    for i,s in enumerate(seq):
        sl=slot(s['bs']);r=[f"{sl['sec']}: {s['phase']} (energy {s['E']:.2f}"+(f" vs target {sl['E']:.2f})" if sl['E'] is not None else ")")]
        if i==0:r.append(f"frame 0 / loop entry: strong {s['phase']} moment mid-action ({sess(s['src'])}); the loop shot is this clip's own lead-in")
        else:
            S,c,why=trans(seq[i-1],s,sl['sec']);r+=why;r.append(f'flow {S:.2f}')
            if s['bs']==8:r.append('hero peak on the drop (strongest burst)')
            if s['bs']==22:r.append('hero speed-ramp payoff')
            if i==len(seq)-1:r.append("loop: opener's own source, ends where frame 0 begins")
        cuts.append(dict(beat=s['bs'],t_out=round(beats[s['bs']],2),src=s['src'],t=s['t'],nb=s['nb'],ramp=s['ramp'],phase=s['phase'],scale=['wide','medium','close',None][s['sc'] if s['sc'] is not None else 3],x=None if s['x'] is None else round(s['x'],2),dir=round(s['dir'],2),reason='; '.join(r)))
    out=dict(SP,id=SP.get('out_id','HR_hype_flow_v5'),shots=[dict(src=s['src'],t=s['t'],bs=s['bs'],nb=s['nb'],ramp=s['ramp'],zoom=s['zoom'],cx=round(s['cx'],3),phase=s['phase'],upscale=s.get('up')) for s in seq],
             sequencing=dict(loop_seam_pick=SEAM,method='narrative v5 (sequence.py: + session/wardrobe continuity, strong frame 0)',flow_before=rep_old,flow_after=rep_new,cuts=cuts,
             pool={p:sum(c['phase']==p for c in P) for p in ['setup','warm-up','build','grind','peak','payoff']},pool_n=len(P)))
    json.dump(out,open(OUT,'w'),indent=1);print(json.dumps(dict(before=rep_old,after=rep_new),indent=1))
    for c in cuts:print(c['beat'],c['src'],c['t'],c['phase'],c['scale'],c['x'],'|',c['reason'])
