"""D3 experimental onset-only full-anchor discovery + <=0.30 s fade backsearch.

First identify 3 temporally consistent frames with the stable event-specific
text/glyph structure (never creates a new kill). Then search at most 0.3s
backwards from THAT observed structure for its first rising high-pass match.
This handles a 10Hz tracker that sees the event well after its actual full box.
Fail closed if no reference frame or pre-fade ramp; retain frozen D2 result.
"""
import cv2,numpy as np

def _highpass(img):
 g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
 return g-cv2.GaussianBlur(g,(7,7),0)

def _ncc(a,b):
 a=a-a.mean();b=b-b.mean()
 denom=float(np.linalg.norm(a)*np.linalg.norm(b))
 return float(np.sum(a*b)/denom) if denom>1e-6 else -1.

def fullanchor_fade_onset(frames,existing_physical,first_obs,reference,competitors=()):
 ref=reference.get('roi')
 if ref is None or len(frames)<10:return None
 h,w=int(first_obs['h']),int(first_obs['w'])
 if not (20<=h<=60 and w>=3*h):return None
 x,y=int(first_obs['x']-1320),int(first_obs['y']-42)
 hp=_highpass(cv2.resize(ref,(w,h),interpolation=cv2.INTER_LINEAR))
 if np.linalg.norm(hp)<25:return None
 # Only the already confirmed track's exact geometry + small registration.
 curve=[]
 for t,img in frames:
  if t>existing_physical+.12:continue
  if x<0 or y<0 or x+w>img.shape[1] or y+h>img.shape[0]:continue
  score=_ncc(hp,_highpass(img[y:y+h,x:x+w]))
  if score>.12:
   for dy,dx in ((0,-4),(0,4),(-2,0),(2,0),(-2,-4),(-2,4),(2,-4),(2,4)):
    xx=x+dx;yy=y+dy
    if xx<0 or yy<0 or xx+w>img.shape[1] or yy+h>img.shape[0]:continue
    score=max(score,_ncc(hp,_highpass(img[yy:yy+h,xx:xx+w])))
  curve.append((t,score))
 if len(curve)<10:return None
 # An actual reference HUD with persistent shape match, rather than a late
 # geometry detector's potentially incorrect 'first full' estimate.
 anchor=None
 for i in range(len(curve)-2):
  t,v=curve[i]
  if v<.70:continue
  if (curve[i+1][1]>=.70 and curve[i+2][1]>=.70 and
      abs(curve[i+2][0]-t-2/60)<.004):
   anchor=i;break
 if anchor is None:return None
 ti=curve[anchor][0]
 loc=[(t,v) for t,v in curve if ti-.30001<=t<=ti+.1001]
 if len(loc)<6:return None
 base=float(np.median([v for _,v in loc[:min(4,len(loc))]]))
 if base>.13:return None
 for j in range(1,len(loc)-3):
  t,v=loc[j];prior=loc[j-1][1]
  if v<.18 or v-prior<.085 or prior>.25:continue
  if (abs(loc[j+2][0]-t-2/60)>.004 or
      loc[j+1][1]<v-.025 or loc[j+2][1]<v-.04):continue
  if max(z[1] for z in loc[j+1:j+6])<.52:continue
  if t>ti or t>=existing_physical-1/60:return None
  return round(t*60)/60.,v,ti
 return None
