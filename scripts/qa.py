# Highlight Reel QA gate (instructions §6), fully automatic. Exit code 0 = all checks pass.
#   run.sh qa.py <job.json>
# job.json: {"out_dir","name","style":"hype|fight|clean","account":"creator|business","seq_spec","render_json",
#            "mux_json","licenses":{"trackA":{"title","artist","page","url"},...},"never_list":[...],"models_dir"}
import sys,os,json,subprocess,hashlib,numpy as np,cv2,mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE);import faces as FC,license_check as LC
J=json.load(open(sys.argv[1]));OD=J['out_dir'];N=J['name'];SP=json.load(open(J['seq_spec']));R=json.loads(open(J['render_json']).read().strip().splitlines()[-1]);MX=json.load(open(J['mux_json']))
MD=J.get('models_dir',os.environ['HR_MODELS']);BPM=float(SP.get('bpm',140));bl=60/BPM
NEVER=[w.lower() for w in J.get('never_list',[])]+['link in bio','guaranteed','lose weight','bet ','odds','shred','fat loss','knockout guaranteed']
f=lambda t:os.path.join(OD,f'{N}_{t}.mp4')
def probe(p):
    o=json.loads(subprocess.check_output(['ffprobe','-v','0','-show_entries','stream=codec_name,profile,width,height,r_frame_rate,pix_fmt,bit_rate,sample_rate,channels:format=duration,bit_rate','-of','json',p]))
    h=open(p,'rb').read(65536);o['faststart']=h.find(b'moov')!=-1 and (h.find(b'mdat')==-1 or h.find(b'moov')<h.find(b'mdat'));return o
files={};vmd5={}
for t in ('trackA','trackB','nomusic'):
    p=probe(f(t));v=[s for s in p['streams'] if s['codec_name']=='h264'][0];a=[s for s in p['streams'] if s['codec_name']=='aac'][0];m=MX[t];vm=int(v['bit_rate'])/1e6
    chk=dict(res=(v['width'],v['height'])==(1080,1920),fps=v['r_frame_rate']=='30/1',codec=v['profile']=='High',pix=v['pix_fmt']=='yuv420p',aac48=a['sample_rate']=='48000',faststart=p['faststart'],mbps_8_12=8<=vm<=12,lufs=abs(m['lufs']+14)<=0.3,tp=m['tp']<=-1.0)
    vmd5[t]=hashlib.md5(subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',f(t),'-map','0:v','-c','copy','-f','data','-'])).hexdigest()
    files[t]=dict(file=f(t),video_mbps=round(vm,2),duration_s=p['format']['duration'],lufs=m['lufs'],tp_dbtp=m['tp'],loudness_iterations=m['loud_iters'],mix=m.get('mix'),checks=chk)
# seam SSIM + contact sheet with glove-filtered faces vs text zones
cap=cv2.VideoCapture(f('nomusic'));Fr=[]
while True:
    ok,fr=cap.read()
    if not ok:break
    Fr.append(fr)
G=[cv2.resize(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),(270,480)).astype(np.float32) for x in Fr]
def ss(x,y):
    C1,C2=6.5025,58.5225;g=lambda z:cv2.GaussianBlur(z,(7,7),1.5);mx,my=g(x),g(y);vx=g(x*x)-mx*mx;vy=g(y*y)-my*my;cxy=g(x*y)-mx*my
    return float((((2*mx*my+C1)*(2*cxy+C2))/((mx*mx+my*my+C1)*(vx+vy+C2))).mean())
seam=ss(G[-1],G[0]);med=float(np.median([ss(G[i],G[i+1]) for i in range(len(G)-1)]))
PL=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(MD,'pose_lite.task')),running_mode=vision.RunningMode.IMAGE,num_poses=3,min_pose_detection_confidence=0.5))
FD=vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=os.path.join(MD,'blaze_face.tflite')),min_detection_confidence=0.5))
n=len(Fr)
# exact text masks: burn the renderer's own ASS onto black and read the pixels (no assumed zones)
ASS=R.get('ass');RD=os.path.dirname(J['render_json'])
if ASS and not os.path.isabs(ASS):ASS=next((p for p in (os.path.join(RD,ASS),os.path.join(RD,os.path.basename(ASS)),os.path.join(OD,os.path.basename(ASS))) if os.path.exists(p)),None)
TM={}
if ASS and os.path.exists(ASS):
    raw=subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',f'color=black:s=1080x1920:r=30:d={n/30:.3f}','-vf',f"subtitles={ASS},select='not(mod(n,7))'",'-vsync','0','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True,check=True).stdout
    A=np.frombuffer(raw,np.uint8).reshape(-1,1920,1080)
    for j,k in enumerate(range(0,n,7)):TM[k]=A[min(j,len(A)-1)]>40
def text_overlap(k,box):
    if k not in TM:return False
    x,y,w,h=box;x0,y0=max(0,x),max(0,y);return TM[k][y0:y+h,x0:x+w].sum()>200
T=[];over=[];gloves=0;ppl={};FB={};IM={};transient=0
KS=list(range(0,n,7))
for k in KS:
    mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(Fr[k],cv2.COLOR_BGR2RGB));pr=PL.detect(mi);fr=FD.detect(mi)
    ppl[k]=sum(1 for p in pr.pose_landmarks if (max(q.y for q in p)-min(q.y for q in p))>0.15);FB[k]=FC.faces(fr,pr,1080,1920)
for j,k in enumerate(KS):
    im=Fr[k].copy();t=k/30
    nb=[f['box'] for kk in KS[max(0,j-1):j]+KS[j+1:j+2] for f in FB[kk] if f['box'] and f['src']=='face']
    for fc in FB[k]:
        gloves+=len(fc['rejected_glove_boxes'])
        for b in fc['rejected_glove_boxes']:cv2.rectangle(im,(b[0],b[1]),(b[0]+b[2],b[1]+b[3]),(255,128,0),3)
        if not fc['box']:continue
        x,y,w,h=fc['box']
        if fc['src']=='face' and not FC.temporal_ok(fc['box'],nb):
            transient+=1;cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),3);continue
        ov=text_overlap(k,fc['box'])
        if ov and fc['src']=='face':over.append(dict(frame=k,t=round(t,2),face_y=[y,y+h]))
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255) if ov else (0,255,0),6)
    im=cv2.resize(im,(216,384));cv2.putText(im,f"{t:.2f}s",(4,378),0,0.6,(0,255,255),2);T.append(im)
cap.release()
while len(T)%10:T.append(np.zeros_like(T[0]))
sheet=os.path.join(OD,f'{N}_contact.png');cv2.imwrite(sheet,np.vstack([np.hstack(T[i:i+10]) for i in range(0,len(T),10)]))
# licenses (track-scoped, see license_check.py)
lic={};
for t,L in J.get('licenses',{}).items():
    r=LC.check(L['page'],J.get('account')=='business');lic[t]=dict(L,license=r['license'],license_url=r.get('url'),evidence=r.get('evidence'),ignored_sitewide=r.get('ignored_sitewide'),commercial_ok=r['commercial_ok'],
        credit=f"\"{L['title']}\" by {L['artist']} ({L['url']}), {r['license']}")
lic_ok=all(lic.get(t,{}).get('license') and (lic[t]['commercial_ok'] or J.get('account')!='business') for t in ('trackA','trackB'))
fr_=SP['sequencing']['flow_after'];texts=[SP.get('hook',''),SP.get('end_line','')]+SP.get('words',[])
never_hits=[w for w in NEVER for x in texts if w in x.lower()]
orig=set(SP.get('orig',{}).values());used={SP['orig'].get(s['src'],s['src']) for s in SP['shots']}
hits_overlay=bool(SP.get('style')=='fight' or J['style']=='fight')
C=dict(files_all_pass=all(all(v['checks'].values()) for v in files.values()),identical_video=len(set(vmd5.values()))==1,seam=seam>=med,flow=fr_['flow']>=80,arc=fr_['arc_rank_corr']>=0.6,
       session_jumps=fr_.get('session_jumps_in_block',0)==0,upscale=max(s['upscale'] for s in R['shots'])<=2.0,no_hits_outside_fight=(J['style']=='fight') or not hits_overlay,
       no_outside_footage=used<=orig,never_list=not never_hits,licenses=lic_ok,text_clear_of_faces=not over,text_outside_bottom_ui=bool(TM) and not any(m[int(1920*0.8):].any() for m in TM.values()),text_mask_source=bool(TM))
Q=dict(reel=N,style=J['style'],account=J.get('account','creator'),checks=C,all_pass=all(C.values()),files=files,seam_ssim=round(seam,3),median_consecutive_ssim=round(med,3),flow=fr_,
       max_upscale=max(s['upscale'] for s in R['shots']),upscale_warn=max(s['upscale'] for s in R['shots'])>1.5,never_list_hits=never_hits,music=lic,contact_sheet=sheet,
       face_text_overlaps=over,glove_face_boxes_rejected=gloves,transient_face_boxes_rejected=transient,people_per_sampled_frame_max=max(ppl.values()),sources_used=sorted(used),cuts=SP['sequencing']['cuts'])
json.dump(Q,open(os.path.join(OD,f'{N}.qa.json'),'w'),indent=1,default=float)
print(json.dumps(dict(all_pass=Q['all_pass'],checks=C,seam=[Q['seam_ssim'],Q['median_consecutive_ssim']],glove_boxes_rejected=gloves,transient_boxes_rejected=transient,face_text_overlaps=len(over),
   loud={t:(v['lufs'],v['tp_dbtp'],len(v['loudness_iterations'])) for t,v in files.items()},licenses={t:v['license'] for t,v in lic.items()}),indent=1))
sys.exit(0 if Q['all_pass'] else 1)
