# prepare.py <job.json> -> writes <work>/in.json and the 720p/30fps analysis proxies (+16 kHz mono wav).
# job.json: {"footage":{"name":"/abs/path.mov",...}, "windows":[{src,a,b,peak}] (optional: auto = whole file in 20 s windows),
#            "hook","emph","words","end_line","bpm","style","id"}
import sys,os,json,subprocess
J=json.load(open(sys.argv[1]));W=os.environ['HR_WORK'];os.makedirs('proxy',exist_ok=True)
def dur(p):return float(subprocess.check_output(['ffprobe','-v','0','-show_entries','format=duration','-of','csv=p=0',p]))
for n,p in J['footage'].items():
    mp4=f'proxy/{n}.mp4'
    if not os.path.exists(mp4):
        # map the first video and the first AAC stream (iPhone spatial audio breaks decoding); honour rotation; CFR 30
        aac=subprocess.run(['ffprobe','-v','0','-select_streams','a','-show_entries','stream=index,codec_name','-of','json',p],capture_output=True,text=True).stdout
        st=[s for s in json.loads(aac or '{}').get('streams',[]) if s.get('codec_name')=='aac'];am=['-map',f"0:{st[0]['index']}"] if st else ['-map','0:a:0?']
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',p,'-map','0:v:0',*am,'-vf',"scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)',fps=30",'-c:v','libx264','-preset','veryfast','-crf','22','-c:a','aac','-ar','48000',mp4],check=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',mp4,'-ac','1','-ar','16000',f'proxy/{n}.wav'],check=True)
win=J.get('windows') or [dict(src=n,a=a,b=min(a+20,dur(p)-1),peak='audio',zoom=1.05) for n,p in J['footage'].items() for a in range(2,int(dur(p))-6,20)]
keys=('bpm','hook','emph','words','end_line','style','music','beam','compare_spec','exclude_compare_moments')
IN=dict({k:J[k] for k in keys if k in J},id=J.get('id','reel'),out_id=J.get('id','reel'),music=J.get('music','none'),orig=J['footage'],windows=win)
json.dump(IN,open('in.json','w'),indent=1);print(f'prepared {len(J["footage"])} sources, {len(win)} windows -> {W}/in.json')
