"""Video-agnostic, HUD-aligned CS2 victim glyph feature. No titles/OCR/counts."""
import cv2,numpy as np

def signature(img):
 if img is None:return None
 h,w=img.shape[:2]
 if h<18 or w < round(h*3.0):return None
 padright=max(6,round(h*.15))
 take=min(w-padright,max(95,round(h*4.6)))
 left=w-padright-take;right=w-padright
 band=img[max(1,round(h*.17)):min(h-1,round(h*.84)),left:right]
 if band.size==0:return None
 b,g,r=[c.astype(np.int16) for c in cv2.split(band)]
 # use differentiated chromatic ink, not white weapon or red HUD border
 blue=(b-r>=35)&(b-g>=6)&(b>=105)&(g>=45)
 gold=(r-b>=38)&(g-b>=29)&(r>=100)&(g>=85)&(np.abs(r-g)<=95)
 # the victim name, regardless of team colour, is right aligned; its right
 # quarter provides colour class without being dominated by another player's tag.
 anchor=int(take*.60)
 b_score=int(np.count_nonzero(blue[:,anchor:]))
 g_score=int(np.count_nonzero(gold[:,anchor:]))
 mask=gold if g_score>b_score else blue
 if int(mask.sum())<max(12,round(h*.35)):return None
 # normalized native pixel crop, including word length: no horizontal scaling
 box=np.zeros((24,256),np.uint8)
 patch=cv2.resize(mask.astype('uint8'),(take,24),interpolation=cv2.INTER_NEAREST)
 box[:,-patch.shape[1]:]=patch
 return box.astype(bool)

def dice(a,b,off=3):
 if a is None or b is None:return 0.
 a=np.asarray(a,np.uint8);b=np.asarray(b,np.uint8)
 mass=int(a.sum())+int(b.sum())
 if not mass:return 0.
 src=np.pad(a,((off,off),(off,off)))
 corr=cv2.matchTemplate(src.astype('float32'),b.astype('float32'),cv2.TM_CCORR)
 return min(1.,float(corr.max()*2/mass))

