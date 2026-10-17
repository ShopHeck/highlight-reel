# Face-aware caption placement. For each text line, gather the face boxes over the line's whole time span from the
# rendered (pre-text) video, with the same glove/pose filter (faces.py) and neighbouring-frame filter the QA uses, plus
# temporal smoothing (boxes from +-2 samples are unioned and padded). Then try, in order:
#   zones in the line's FIXED order (hook, instructions §4.2: top safe area -> upper-third left -> upper-third right ->
#   mid-frame above centre), first at 100% font, then shrunk 95/90/85%, then (last resort) shifted by one beat.
# Hard bans: never over a face; never in the bottom 20% of the frame (y >= 1536, platform UI). If nothing fits, ok=False
# and the QA gate fails the reel (never delivered silently).
# The text box is MEASURED by burning the real ASS line onto black (at its largest pop-in scale), not estimated.
import os,subprocess,tempfile,numpy as np,cv2,mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions
import faces as FC
H=1920;UI_Y=int(H*0.8)   # bottom 20% = platform UI zone, text banned
# zone = (name, ASS alignment, x, y, narrow) ; narrow zones break the text so it fits one half of the frame
# top safe area starts at y=160 (below the platform top bar); upper third = y 160-640; mid-frame above centre ends at y=930
HOOK_ZONES=[('top',8,540,160,False),('upper-third-left',7,50,160,True),('upper-third-right',9,1030,160,True),('mid-above-centre',2,540,930,False)]
WORD_ZONES=[('lower-middle',5,540,1180,False),('top',8,540,160,False),('mid-above-centre',2,540,930,False),('lower',5,540,1380,False)]
END_ZONES=HOOK_ZONES
PAD=24
def detect(video,models,step=3,cuts=None):
    """cuts: output frame indices where shots start; smoothing never crosses a cut (faces jump at cuts)"""
    PL=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(models,'pose_lite.task')),running_mode=vision.RunningMode.IMAGE,num_poses=3,min_pose_detection_confidence=0.5))
    FD=vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=os.path.join(models,'blaze_face.tflite')),min_detection_confidence=0.5))
    cap=cv2.VideoCapture(video);k=0;raw={}
    while True:
        ok=cap.grab()
        if not ok:break
        if k%step==0:
            ok,f=cap.retrieve();mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            raw[k]=[x['box'] for x in FC.faces(FD.detect(mi),PL.detect(mi),f.shape[1],f.shape[0]) if x['box'] and x['src']=='face']
        k+=1
    cap.release();K=sorted(raw);B={}
    for j,k in enumerate(K):
        nb=[b for kk in K[max(0,j-1):j]+K[j+1:j+2] for b in raw[kk]]
        B[k]=[b for b in raw[k] if FC.temporal_ok(b,nb)]
    S={}
    import bisect
    shot=lambda k:bisect.bisect_right(sorted(cuts),k) if cuts else 0
    for j,k in enumerate(K):   # smoothing: union of filtered boxes over +-2 samples inside the same shot
        S[k]=[b for kk in K[max(0,j-2):j+3] if shot(kk)==shot(k) for b in B[kk]]
    return S
_M={}
def measure(header,style,text,tags):
    key=(style,text,tags)
    if key in _M:return _M[key]
    d=tempfile.mkdtemp();p=os.path.join(d,'m.ass')
    open(p,'w').write(header+f"\nDialogue: 0,0:00:00.00,0:00:01.00,{style},,0,0,0,,{{{tags}}}{text}\n")
    raw=subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=black:s=1080x1920:r=1:d=1','-vf',f'subtitles={p}','-frames:v','1','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True,check=True).stdout
    a=np.frombuffer(raw,np.uint8).reshape(1920,1080)>40;ys,xs=np.where(a)
    _M[key]=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None;return _M[key]
def hits(box,boxes):
    if box is None:return False
    x0,y0,x1,y1=box
    return any(x0-PAD<b[0]+b[2] and x1+PAD>b[0] and y0-PAD<b[1]+b[3] and y1+PAD>b[1] for b in boxes)
def _wrap(header,style,text,fs,m,maxw=470):
    """greedy line breaks so the measured width stays inside one half of the frame"""
    w=text.split(' ');lines=[];cur=''
    for x in w:
        t=(cur+' '+x).strip();b=measure(header,style,t,f"\\an5\\pos(540,960)\\fs{fs}\\fscx{m}\\fscy{m}")
        if cur and b and b[2]-b[0]>maxw:lines.append(cur);cur=x
        else:cur=t
    lines.append(cur);return '\\N'.join(lines)
def place(header,line,S,fps,beat,t_max):
    """line: dict(style,text,spans=[(st,en,scale%)...],zones=[...],fs, shift_ok, shift_mode 'both'|'start'|'end')
    returns dict(zone,pos=ASS tags,text,fs,scale,shift,ok)"""
    zones=line['zones'];shifts=[0.0]+([beat] if line.get('shift_ok',True) else []);md=line.get('shift_mode','both')
    for sh in shifts:
        sp=[]
        for j,(a,b,m) in enumerate(line['spans']):
            if md=='start':a=a+sh
            elif md=='end':b=b-sh if j==line.get('end_idx',0) else b
            else:a,b=(a+sh,min(b+sh,t_max)) if sh else (a,b)
            sp.append((a,b,m))
        if any(b-a<0.1 for a,b,_ in sp):continue
        for sc in (1.0,0.95,0.9,0.85):
            fs=int(round(line['fs']*sc))
            for zn,an,x,y,narrow in zones:
                mx=max(m for _,_,m in sp);txt=_wrap(header,line['style'],line['text'],fs,mx) if narrow else line['text']
                pos=f"\\an{an}\\pos({x},{y})\\fs{fs}";bad=False
                for a,e,m in sp:
                    tb=measure(header,line['style'],txt,pos+f"\\fscx{m}\\fscy{m}")
                    if tb is None or tb[3]>=UI_Y or tb[0]<0 or tb[2]>1079 or tb[1]<0:bad=True;break
                    boxes=[b for k,bb in S.items() if a-0.05<=k/fps<=e+0.05 for b in bb]
                    if hits(tb,boxes):bad=True;break
                if not bad:return dict(zone=zn,pos=pos,text=txt,fs=fs,scale=sc,shift=sh,ok=True)
    zn,an,x,y,_=zones[0];return dict(zone=zn,pos=f"\\an{an}\\pos({x},{y})\\fs{line['fs']}",text=line['text'],fs=line['fs'],scale=1.0,shift=0.0,ok=False)
