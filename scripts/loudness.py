# Shared loudness loop for every Highlight Reel deliverable (render_flow.py, mux.py, music/real/mux_real.py).
# Target: -14 LUFS integrated (+-0.3) and true peak <= -1 dBTP, MEASURED ON THE FINAL ENCODED AAC FILE (ebur128).
# The loop moves gain in BOTH directions and also moves the limiter ceiling in both directions:
#   - TP too hot            -> lower the ceiling
#   - too quiet and TP has headroom -> RAISE the ceiling (a fixed low ceiling was what stalled the old loop at -14.8 LUFS:
#     extra gain only went into the limiter) and raise gain
#   - too loud              -> lower gain
# Gain steps use the measured dLUFS/dGain slope (secant), so a limiter that eats part of each step is compensated.
import subprocess
def meas(f):
    m=subprocess.run(['ffmpeg','-nostdin','-nostats','-i',f,'-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True).stderr.splitlines()
    I=float([l for l in m if l.strip().startswith('I:')][-1].split()[1]);P=float([l for l in m if l.strip().startswith('Peak:')][-1].split()[1])
    return I,P
def mux(video,audio,out,target=-14.0,tol=0.3,tp_max=-1.0,lim0=-2.0,lim_max=-1.2,iters=14,log=None):
    I0,_=meas(audio);G=target-I0;LIM=lim0;hist=[];slope=1.0
    for it in range(iters):
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',video,'-i',audio,'-map','0:v','-map','1:a','-c:v','copy',
            '-af',f'volume={G:.2f}dB,alimiter=limit={10**(LIM/20):.4f}:attack=2:release=60:level=disabled',
            '-c:a','aac','-b:a','192k','-ar','48000','-shortest','-movflags','+faststart',out],check=True)
        I,P=meas(out);hist.append((round(G,2),round(LIM,2),I,P))
        if log:log(f'loud it{it}: gain {G:+.2f} dB ceiling {LIM:.2f} dBFS -> {I} LUFS {P} dBTP')
        if abs(I-target)<=tol and P<=tp_max:return dict(lufs=I,tp=P,loud=hist,ok=True)
        err=target-I
        if P>tp_max:LIM-=(P-tp_max)+0.2                                   # too hot: pull the ceiling down
        elif err>0 and P<tp_max-0.4 and LIM<lim_max:                       # too quiet with peak headroom: open the ceiling
            LIM=min(lim_max,LIM+min(2.0,(tp_max-0.3)-P))
        if len(hist)>=2 and abs(hist[-1][0]-hist[-2][0])>0.05 and hist[-1][1]==hist[-2][1]:
            slope=min(1.5,max(0.25,(hist[-1][2]-hist[-2][2])/(hist[-1][0]-hist[-2][0])))
        G+=max(-8.0,min(8.0,err/slope))                                    # raise OR lower gain
    return dict(lufs=I,tp=P,loud=hist,ok=False)
