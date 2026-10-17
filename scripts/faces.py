# Face boxes with glove/mitt false-positive rejection.
# BlazeFace fires on boxing gloves and pads. A face box is REJECTED when a pose wrist/hand keypoint
# (15-22: wrists, pinkies, index, thumbs) lies inside it while the pose's own head keypoints do not.
# If every face box is rejected (or none was found) and a pose is present, fall back to the pose head box
# (from nose/eyes/ears keypoints 0-10), or the pose box when the head points are not visible.
HAND=range(15,23);HEAD=range(0,11)
def _in(p,b,pad=0.0):
    x0,y0,x1,y1=b;w,h=x1-x0,y1-y0
    return x0-pad*w<=p[0]<=x1+pad*w and y0-pad*h<=p[1]<=y1+pad*h
def faces(face_res,pose_res,W,H,vis=0.3):
    """returns list of dict(box=[x,y,w,h] px, src='face'|'pose_head'|'pose_box', rejected=[...])"""
    poses=[[(q.x*W,q.y*H,(q.visibility if q.visibility is not None else 1.0)) for q in lm] for lm in (pose_res.pose_landmarks if pose_res else [])]
    keep=[];rej=[]
    for d in (face_res.detections if face_res else []):
        b=d.bounding_box;bb=(b.origin_x,b.origin_y,b.origin_x+b.width,b.origin_y+b.height)
        glove=False
        for P in poses:
            hand_in=any(P[i][2]>vis and _in(P[i],bb,0.1) for i in HAND)
            head_in=any(P[i][2]>vis and _in(P[i],bb) for i in HEAD)
            if hand_in and not head_in:glove=True
            # box not on this pose's head but next to its wrist (gloves/pads held off the face)
            hv=[P[i] for i in HEAD if P[i][2]>vis];wv=[P[i] for i in (15,16) if P[i][2]>vis];sz=max(bb[2]-bb[0],bb[3]-bb[1])
            if hv and wv and not head_in and any(_in(w,bb,1.0) for w in wv) and not any(_in(h,bb,0.25) for h in hv):glove=True
        (rej if glove else keep).append([b.origin_x,b.origin_y,b.width,b.height])
    out=[dict(box=k,src='face') for k in keep]
    if not out:
        for P in poses:
            hd=[p for i,p in enumerate(P) if i in HEAD and p[2]>vis];pts=hd or [p for p in P if p[2]>vis]
            if not pts:continue
            xs=[p[0] for p in pts];ys=[p[1] for p in pts];pad=0.35*(max(xs)-min(xs)+1) if hd else 0
            x0,x1=min(xs)-pad,max(xs)+pad;y0=min(ys)-pad*1.4;y1=max(ys)+pad*1.2
            out.append(dict(box=[int(x0),int(y0),int(x1-x0),int(y1-y0)],src='pose_head' if hd else 'pose_box'))
    for o in out:o['rejected_glove_boxes']=rej
    return out if out else ([dict(box=None,src=None,rejected_glove_boxes=rej)] if rej else [])
def temporal_ok(box,neighbours):
    """A real face moves continuously: keep a box only if a face box in an adjacent sample (about 0.23 s away) sits
    within one box size of it. Transient one-sample boxes (a motion-blurred mitt or glove) fail this test."""
    if not neighbours:return True
    cx,cy=box[0]+box[2]/2,box[1]+box[3]/2;s=max(box[2],box[3])
    return any(abs(cx-(b[0]+b[2]/2))<=s and abs(cy-(b[1]+b[3]/2))<=s for b in neighbours)
