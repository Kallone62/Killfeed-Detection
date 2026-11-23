"""CS2 physical HUD attacker-side consensus. Never inspect victim/title/expected count.

A red four-sided frame only establishes an OWN-COLOURED HUD row; it does not
prove the attacker is the target player (spectator/camera can change). The
attacker text is estimated from native pixels, and majority across independent
source events is used as clip-local owner. OCR has *veto-only* authority when
it repeatedly identifies a DISTINCT, non-owner attacker. Uncertain OCR cannot
create, merge, retime or delete a physical event.
"""
from __future__ import annotations
import cv2,numpy as np,subprocess,re,collections


def attacker_region(img):
 """Text left of the first white weapon, detected by HUD glyph geometry.

 A fixed 4.2*row-height crop includes the white gun and changes OCR "Descrip"
 into "teche" on some colour schemes. The first coherent neutral/white gun
 run is a natural right boundary; this also handles variable-length attackers
 and optional assist labels. There is no clip-specific parameter.
 """
 if img is None:return None
 h,w=img.shape[:2]
 if h<18 or w<round(h*3):return None
 x0=max(3,round(h*.12))
 top=max(2,round(h*.17));bottom=min(h-2,round(h*.84))
 bgr=img[top:bottom,:]
 if bgr.size==0:return None
 gray_channels=bgr.astype(np.int16)
 white=(gray_channels.min(axis=2)>=185)&((gray_channels.max(axis=2)-gray_channels.min(axis=2))<38)
 counts=white.sum(axis=0)
 hits=np.flatnonzero(counts>=max(3,round(h*.16)))
 runs=[]
 if len(hits):
  st=pr=int(hits[0])
  for xx in hits[1:]:
   xx=int(xx)
   if xx>pr+1:
    runs.append((st,pr));st=xx
   pr=xx
  runs.append((st,pr))
 weapon_starts=[left for left,right in runs
                if right-left+1>=round(h*.35) and left>=round(h*.65)]
 x1=(min(weapon_starts)-2 if weapon_starts else min(w-5,round(h*3.6)))
 if x1-x0<round(h*.85):x1=min(w-5,round(h*3.6))
 return img[top:bottom,x0:max(x0+1,x1)].copy()


def ocr_attacker(img,tesseract='tesseract'):
 area=attacker_region(img)
 if area is None:return ''
 gray=cv2.cvtColor(area,cv2.COLOR_BGR2GRAY)
 big=cv2.resize(gray,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
 mask=cv2.threshold(big,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
 mask=cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
 success,data=cv2.imencode('.png',mask)
 if not success:return ''
 args=[tesseract,'stdin','stdout','--psm','7','-l','eng','-c',
       'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-_']
 try:
  cp=subprocess.run(args,input=data.tobytes(),capture_output=True,timeout=35,
                    env={**__import__("os").environ,"OMP_THREAD_LIMIT":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
  if cp.returncode:return ''
  initial=cp.stdout.decode('utf-8','replace').strip().replace('\n',' ')
  # A one-letter psm7 fragment is inconclusive, not negative evidence.
  # Only then retry with another deterministic single-line segmentation.
  if len(_prefix(initial))>=5:return initial
  retry=list(args);retry[4]='13'
  cp2=subprocess.run(retry,input=data.tobytes(),capture_output=True,timeout=35,
                    env={**__import__('os').environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  if cp2.returncode:return initial
  alt=cp2.stdout.decode('utf-8','replace').strip().replace('\n',' ')
  return alt if len(_prefix(alt))>=5 else initial
 except (OSError,subprocess.TimeoutExpired):return ''


def edit_dist(a,b):
 if a==b:return 0
 p=list(range(len(b)+1))
 for i,ca in enumerate(a,1):
  q=[i]
  for j,cb in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+int(ca!=cb)))
  p=q
 return p[-1]


def _prefix(text):
 # Parse the ATTACKER before a '+' assist separator. Small icon fragments
 # separated by whitespace must not be glued to a complete player token.
 # E.g. "Hp Descrip+BOT_SWATT" means Descrip, not HpDescrip; an unrelated
 # full player name (WobblyAim) remains a distinct OTHER candidate.
 before_assist=text.split('+',1)[0].lower()
 tokens=re.findall(r'[a-z]+',before_assist)
 return max(tokens,key=len)[:8] if tokens else ''


def attacker_glyph_signature(img):
 """Canonical binary attacker-name ink + text hue from native PNG pixels.

 The local brightness contrast isolates both yellow OWN and blue OTHER text;
 the previous yellow-only mask mislabeled blue text as background.
 """
 area=attacker_region(img)
 if area is None:return None
 hsv=cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
 h,s,v=cv2.split(hsv)
 text=(v.astype(np.int16)>=max(115.,float(np.median(v))+45)).astype('uint8')
 yy,xx=np.where(text)
 if len(xx)<15:return None
 crop=text[max(0,int(yy.min())-1):min(text.shape[0],int(yy.max())+2),int(xx.min()):]
 if crop.shape[0]<3 or crop.shape[1]<8:return None
 scale=24/float(crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
 result=np.zeros((24,100),dtype=np.uint8)
 result[:min(24,crop.shape[0]),:min(100,crop.shape[1])]=crop[:24,:100]
 # Hue is secondary contrast evidence (not sufficient without distinct OCR).
 col=h[text>0]
 hue=float(np.median(col)) if len(col) else None
 return (result,hue)


def _glyph_dice(a,b):
 if a is None or b is None:return None
 aa,ah=a;bb,bh=b
 return float(2.*np.count_nonzero(aa & bb)/max(1,np.count_nonzero(aa)+np.count_nonzero(bb)))


def _glyph_is_different(candidate, owner_templates):
 if candidate is None or not owner_templates:return False,None
 same=max(_glyph_dice(candidate,ref) for ref in owner_templates)
 h=candidate[1]
 own_hues=[r[1] for r in owner_templates if r[1] is not None]
 hue_gap=None if h is None or not own_hues else float(abs(h-float(np.median(own_hues))))
 if hue_gap is not None:hue_gap=min(hue_gap,180.-hue_gap)
 # Strong grayscale difference OR clearly different name + ink colour.
 visually_different=(same<.63 or (same<.68 and hue_gap is not None and hue_gap>25))
 return visually_different,{'max_name_dice':round(same,4),'hue_delta':round(hue_gap,2) if hue_gap is not None else None}


def assess_ownership(event_images,tesseract='tesseract',owner_name=''):
 """Input independent source PNGs PER PHYSICAL event, up to two samples.
 Output (decisions, evidence) with no video-specific name, threshold, or count.
 """
 reads=[[ocr_attacker(im,tesseract) for im in images[:2]] for images in event_images]
 keys=[[_prefix(s) for s in rr if len(_prefix(s))>=5] for rr in reads]
 # The red-owned frame is OWN by default. OCR is only a strong negative veto.
 # Do not let one noisy read or an unrecognised text remove a real kill.
 votes=[arr[0] for arr in keys if arr]
 cnt=collections.Counter(votes)
 cand=list(cnt)
 scores={x:sum(1 for k in votes if edit_dist(x[:7],k[:7])<=2) for x in cand}
 canonical=(_prefix(owner_name) if owner_name else
            (max(cand,key=lambda x:(scores[x],cnt[x],len(x))) if cand else ''))
 owner_events=[i for i,arr in enumerate(keys) if any(edit_dist(canonical[:7],x[:7])<=2 for x in arr)] if canonical else []
 # Reference ink is learned from confirmed OCR OWN events in this video.
 # Diverse source frames avoid a single crop artefact becoming the template.
 owner_templates=[]
 for i in owner_events:
  for img in event_images[i][:2]:
   sig=attacker_glyph_signature(img)
   if sig is not None:owner_templates.append(sig)
 decisions=[];reasons=[];ink_evidence=[]
 for i,arr in enumerate(keys):
  if canonical and any(edit_dist(canonical[:7],x[:7])<=2 for x in arr):
   decisions.append('OWN');reasons.append('ATTACKER_MATCH');ink_evidence.append(None);continue
  distinct=[x for x in arr if canonical and len(x)>=6 and
            edit_dist(canonical[:7],x[:7])>=4]
  strong_ocr=(len(distinct)>=2 and edit_dist(distinct[0][:7],distinct[1][:7])<=2)
  ink_scores=[];ink_distinct=[]
  for img in event_images[i][:2]:
   visually_different,diagnostic=_glyph_is_different(attacker_glyph_signature(img),owner_templates)
   ink_scores.append(diagnostic)
   ink_distinct.append(visually_different)
  # Distinct OCR and visible ink difference must BOTH recur in two source frames.
  strong_glyph=len(ink_distinct)>=2 and all(ink_distinct[:2])
  ink_evidence.append(ink_scores)
  if strong_ocr and strong_glyph:
   decisions.append('OTHER');reasons.append('TWO_DIFFERENT_OCR_READS_AND_DISTINCT_ATTACKER_GLYPH')
  else:
   decisions.append('OWN');reasons.append('DEFAULT_OWN_INSUFFICIENT_OTHER_EVIDENCE')
 return decisions,{'owner_prefix':canonical,'reads':reads,'event_prefixes':keys,
                   'reasons':reasons,'owner_reference_count':len(owner_templates),
                   'glyph_dice_to_owner':ink_evidence}

