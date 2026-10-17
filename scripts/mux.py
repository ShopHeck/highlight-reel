# CANONICAL music path for Highlight Reel deliverables.
#   run.sh mux.py <render_nomusic_sfx.mp4> <out_dir> <name> [trackA_bed.wav] [trackB_bed.wav]
# Input: the render from render_flow.py with music "none" (video + SFX layer: whooshes, hits, riser, source impacts).
# Output: <name>_trackA.mp4, <name>_trackB.mp4, <name>_nomusic.mp4 with IDENTICAL video (stream copy) and
#   music = licensed bed MIXED WITH the SFX layer (bed 0.85, SFX 0.35 after peak-normalising each), never replacing it.
# Loudness: loudness.mux on the final AAC (-14 +-0.3 LUFS, TP <= -1 dBTP), gain and ceiling move both ways.
# Beds must already be trimmed/stretched so their drop lands on the visual drop (music/real/build_real.py).
import sys,os,json,numpy as np,soundfile as sf,subprocess
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import loudness
def mix_bed_sfx(sfx,bed,n):
    s=sfx/(np.abs(sfx).max()+1e-9)
    if bed is None:return s.copy()
    b=bed if bed.ndim==2 else np.stack([bed,bed],1)
    b=b[:n] if len(b)>=n else np.pad(b,((0,n-len(b)),(0,0)));b=b/(np.abs(b).max()+1e-9)
    return 0.85*b+0.35*s
def run(V,OD,N,beds):
    os.makedirs(OD,exist_ok=True);tmp=os.path.join(OD,f'.{N}_sfx.wav')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',V,'-vn','-ac','2','-ar','48000','-c:a','pcm_f32le',tmp],check=True)
    sfx,sr=sf.read(tmp);n=len(sfx);res={}
    for tag,bed in beds:
        b=None
        if bed:
            b,bsr=sf.read(bed)
            if bsr!=sr:
                import librosa;b=librosa.resample(b.T,orig_sr=bsr,target_sr=sr).T
        mix=mix_bed_sfx(sfx,b,n);mw=os.path.join(OD,f'.{N}_mix_{tag}.wav');sf.write(mw,mix/np.abs(mix).max()*0.5,sr,subtype='FLOAT')
        fo=os.path.join(OD,f'{N}_{tag}.mp4');r=loudness.mux(V,mw,fo);os.remove(mw)
        res[tag]=dict(file=fo,bed=bed,mix='bed 0.85 + SFX 0.35' if bed else 'SFX only',lufs=r['lufs'],tp=r['tp'],loud_iters=r['loud'],loud_ok=r['ok'])
    os.remove(tmp);json.dump(res,open(os.path.join(OD,f'{N}.mux.json'),'w'),indent=1);return res
if __name__=='__main__':
    V,OD,N=sys.argv[1:4];bd=sys.argv[4:6]
    beds=[(f'track{"AB"[i]}',os.path.abspath(b)) for i,b in enumerate(bd)]+[('nomusic',None)]
    print(json.dumps(run(os.path.abspath(V),os.path.abspath(OD),N,beds),indent=1))
