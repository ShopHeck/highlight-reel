# fraction of 1 fps frames with two similarly sized, large people (pad holder + striker / sparring partner)
import sys,json,numpy as np,cv2,mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions
def two_person(path,step=1.0):
    o=vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=__import__('os').path.join(__import__('os').environ['HR_MODELS'],'pose_lite.task')),running_mode=vision.RunningMode.IMAGE,num_poses=3,min_pose_detection_confidence=0.5)
    L=vision.PoseLandmarker.create_from_options(o);cap=cv2.VideoCapture(path);dur=cap.get(7)/(cap.get(5) or 30);hits=[];t=0
    while t<dur:
        cap.set(0,t*1000);ok,f=cap.read();t+=step
        if not ok:break
        f=cv2.resize(f,(int(f.shape[1]*360/f.shape[0]),360));r=L.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
        h=sorted([max(q.y for q in p)-min(q.y for q in p) for p in r.pose_landmarks],reverse=True)
        hits.append(len(h)>=2 and h[1]>=0.35 and h[1]/h[0]>=0.6)
    return float(np.mean(hits)) if hits else 0.0
if __name__=='__main__':
    out={}
    for p in sys.argv[1:]:out[p]=two_person(p);print(p,round(out[p],2),flush=True)
