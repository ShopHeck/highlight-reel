# Narrative sequencing: per-moment features from the 720p analysis proxies (the owner's submitted footage only).
# track(src,t): 10 fps, t-1.6..t+2.2 s: subject box (mediapipe pose_lite, largest person), wrist speed,
# Farneback flow inside the subject box (mean dx = screen direction, magnitude = energy), whole-frame motion, thumbnail.
import os,json,numpy as np,cv2,mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions
CACHE='flow/feat_cache.json';_C=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_L=None
def _lmk():
    global _L
    if _L is None:
        o=vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(os.environ['HR_MODELS'],'pose_lite.task')),running_mode=vision.RunningMode.IMAGE,num_poses=2,min_pose_detection_confidence=0.4)
        _L=vision.PoseLandmarker.create_from_options(o)
    return _L
def save():json.dump(_C,open(CACHE,'w'))
def track(src,t,a=-1.6,b=2.2,step=0.1):
    key=f'{src}@{t:.2f}'
    if key in _C:return _C[key]
    cap=cv2.VideoCapture(f'proxy/{src}.mp4');fps=cap.get(5) or 30;n=int(cap.get(7));H=W=None;R=[];pg=None;pbox=None
    ts=np.arange(t+a,t+b+1e-6,step)
    f0=max(0,int(round(ts[0]*fps)));cap.set(1,f0);cur=f0-1;frm=None
    for tt in ts:
        fi=int(round(tt*fps))
        if fi<0 or fi>=n:R.append(None);pg=None;continue
        ok=True
        while cur<fi and ok:ok,frm=cap.read();cur+=1
        f=frm
        if not ok or f is None:R.append(None);pg=None;continue
        H,W=f.shape[:2];g=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(180,int(180*H/W)))
        r=_lmk().detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)));best=None;ba=0
        for p in r.pose_landmarks:
            xs=[q.x for q in p if q.visibility>0.3];ys=[q.y for q in p if q.visibility>0.3]
            if len(xs)<5:continue
            ar=(max(xs)-min(xs))*(max(ys)-min(ys))
            if ar>ba:ba=ar;best=p
        rec=dict(t=float(tt))
        if best:
            v=lambda i:best[i].visibility>0.5
            xs=[q.x for q in best if q.visibility>0.3];ys=[q.y for q in best if q.visibility>0.3]
            rec.update(x0=min(xs),x1=max(xs),y0=min(ys),y1=max(ys),cx=float(np.median([best[i].x for i in (0,11,12,23,24)])),
                       ank=bool(v(27) or v(28)),hip=bool(v(23) or v(24)),wr=[[best[15].x,best[15].y],[best[16].x,best[16].y]])
            pbox=rec
        if pg is not None:
            fl=cv2.calcOpticalFlowFarneback(pg,g,None,0.5,3,15,3,5,1.2,0);h,w=g.shape
            bx=pbox or dict(x0=0.2,x1=0.8,y0=0.1,y1=0.9);xa,xb=int(np.clip(bx['x0'],0,1)*w),int(np.clip(bx['x1'],0,1)*w)+1;ya,yb=int(np.clip(bx['y0'],0,1)*h),int(np.clip(bx['y1'],0,1)*h)+1
            sub=fl[ya:yb,xa:xb].reshape(-1,2);bg=np.median(fl.reshape(-1,2),0)   # subtract camera (median) motion
            sub=sub-bg;mag=np.linalg.norm(sub,axis=1);m=mag>np.percentile(mag,70)
            rec.update(mag=float(mag.mean()),dx=float(sub[m,0].mean()) if m.any() else 0.0,cam=float(np.linalg.norm(bg)))
        pg=g;R.append(rec)
    th=None
    cap.set(1,int(round(t*fps)));ok,f=cap.read()
    if ok:th=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(16,28)).astype(float).ravel().tolist()
    cap.release();print('track',key,flush=True);_C[key]=dict(src=src,t=t,W=W,H=H,rec=R,thumb=th);save();return _C[key]
def at(tr,t0,t1,k):
    v=[r[k] for r in tr['rec'] if r and k in r and t0<=r['t']<=t1]
    return float(np.mean(v)) if v else None
def wrist_speed(tr,t0,t1):
    P=[(r['t'],np.array(r['wr'])) for r in tr['rec'] if r and 'wr' in r and t0-0.1<=r['t']<=t1+0.1]
    if len(P)<2:return 0.0,0.0
    sp=[];dx=[]
    for (ta,a),(tb,b) in zip(P[:-1],P[1:]):
        d=(b-a)/(tb-ta);i=int(np.argmax(np.linalg.norm(d,axis=1)));sp.append(np.linalg.norm(d[i]));dx.append(d[i][0])
    return float(np.mean(sp)),float(np.mean(dx))
