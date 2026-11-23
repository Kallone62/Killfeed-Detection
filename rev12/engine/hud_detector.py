"""Source-only four-sided CS2 HUD detector. No title/OCR/GT inputs."""
import cv2,numpy as np,subprocess
from functools import lru_cache
from pathlib import Path
FPS=10; X0=1320; Y0=42; W=600; H=310
ROW_PITCH=round(1080*.0305)

@lru_cache(maxsize=64)
def _content_letterbox_rect(filename, ffmpeg):
    """Normalize EXPLICIT symmetric black letterbox bars, not ordinary dark gameplay.

    A 90%-scaled HUD inside a 1920x1080 black pad is geometrically identical
    to the original after the content area is recropped. Inspect an early
    independent source frame once; ignore scenes without symmetric wide bars.
    """
    probe=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-threads','1',
           '-ss','4.000','-i',str(filename),'-frames:v','1',
           '-vf','scale=1920:1080:flags=fast_bilinear',
           '-f','rawvideo','-pix_fmt','gray','-']
    try:
        z=subprocess.run(probe,capture_output=True,timeout=18)
        if z.returncode!=0 or len(z.stdout)!=1920*1080:return None
        gray=np.frombuffer(z.stdout,dtype=np.uint8).reshape(1080,1920)
        c=(gray[216:864]>10).mean(axis=0)
        r=(gray[:,384:1536]>10).mean(axis=1)
        cx=np.flatnonzero(c>=.04);cy=np.flatnonzero(r>=.04)
        if not len(cx) or not len(cy):return None
        left=int(cx[0]);right=1919-int(cx[-1]);top=int(cy[0]);bottom=1079-int(cy[-1])
        if not (left>=50 and top>=27 and abs(left-right)<=6 and abs(top-bottom)<=6):return None
        w=1920-left-right;h=1080-top-bottom
        if w<1500 or h<800 or abs(w/h-16/9)>.025:return None
        # Chroma subsampling requires even crop coordinates and extents.
        x=left+(left%2);y=top+(top%2)
        cw=1920-x-right;ch=1080-y-bottom
        cw-=cw%2;ch-=ch%2
        return (cw,ch,x,y)
    except (OSError,subprocess.TimeoutExpired,ValueError):return None


def source_frames(filename, fps=FPS, *, start=0., end=None, ffmpeg='ffmpeg'):
    """Decode source HUD exactly once and fail closed on truncated FFmpeg output.

    Overcommitted FFmpeg decoder/filter threads previously yielded incomplete
    observation streams without being detected. A short stream cannot be
    safely treated as evidence that no new kill appeared in the missing time.
    """
    if end is not None and end <= start:
        raise ValueError('Invalid source time range')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin',
         '-threads','1','-filter_threads','1','-ss',f'{start:.6f}',
         '-i',str(filename)]
    if end is not None:
        cmd += ['-t',f'{end-start:.6f}']
    box=_content_letterbox_rect(str(filename),str(ffmpeg))
    normal=f'fps={fps},scale=1920:1080:flags=lanczos'
    if box:
        cw,ch,x,y=box
        normal+=f',crop={cw}:{ch}:{x}:{y},scale=1920:1080:flags=lanczos'
    cmd += ['-vf',normal+f',crop={W}:{H}:{X0}:{Y0}',
            '-threads','1','-f','rawvideo','-pix_fmt','bgr24','-']
    process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=10**7)
    i=0;completed=False
    try:
        while True:
            data=process.stdout.read(W*H*3)
            if not data:
                completed=True
                break
            if len(data)!=W*H*3:
                raise RuntimeError('FFmpeg yielded a truncated raw BGR frame')
            yield round(start+i/fps,6),np.frombuffer(data,dtype='uint8').reshape(H,W,3).copy()
            i+=1
    finally:
        if process.stdout:process.stdout.close()
        if not completed and process.poll() is None:process.terminate()
        rc=process.wait()
        error=process.stderr.read().decode('utf8','replace') if process.stderr else ''
        if process.stderr:process.stderr.close()
        if completed:
            if rc!=0:
                raise RuntimeError('FFmpeg source decode failed rc=%s: %s'%(rc,error[-1100:]))
            if end is not None and i < int((end-start)*fps)-2:
                raise RuntimeError('Incomplete source frame sequence: got %s, expected about %s'
                                   %(i,round((end-start)*fps)))


def bmask(image):
 b,g,r=(a.astype(np.int16) for a in cv2.split(image))
 # Pink/flash or scene-tinted red HUD outline retains red dominance, but
 # the original saturated-red-only ratio can erase its complete top edge.
 # Geometry verifies all four sides, preventing ordinary scenery from qualifying.
 return (r>108)&(r>g*1.18)&(r>b*1.30)

def strict_bmask(image):
 b,g,r=(a.astype(np.int16) for a in cv2.split(image))
 return (r>125)&(r>g*1.85)&(r>b*1.50)&(g<115)

def dim_red_frame_mask(image):
    # A photometrically darkened HUD retains relative red dominance even when
    # absolute red is below the strict saturation gate. Require the SAME full
    # four-edge geometry on both the horizontal and vertical sides.
    b,g,r=(a.astype(np.int16) for a in cv2.split(image))
    # Dim mask is the complement of the *absolute* strong-red band.
    # Without the upper bound, bright HUD tails can recreate an old event.
    return (r>70)&(r<=125)&(r>g*1.70)&(r>b*1.50)&(g<115)

def segments(prof, threshold, gap=0):
 a=prof>=threshold
 if gap:
  # no opening, only close one-pixel holes in support
  for i in range(1,len(a)-1):
   if not a[i] and a[i-1] and a[i+1]:a[i]=True
 idx=np.flatnonzero(np.diff(np.r_[False,a,False].astype('int8')))
 return list(zip(idx[::2],idx[1::2]-1))

def _detect_with_mask(img,red,hard):
 # Pair soft horizontal strokes, but choose the x position of the left edge
 # from the independent SATURATED vertical side. Scaling can blur horizontal
 # top/bottom edges and merge a red scene wall into the apparent left extent.
 # Hard vertical evidence + dark interior reject cross-row grey rectangles.
 lines=[]
 for y in range(round(ROW_PITCH*.55),min(H-round(ROW_PITCH*.55),round(ROW_PITCH*7.1))):
  col=red[y,:W-6].astype(np.uint8)
  col=cv2.morphologyEx(col.reshape(1,-1),cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))[0]
  spans=segments(col,1)
  candidate=[(q-p+1,p,q) for p,q in spans if q-p+1>=round(ROW_PITCH*3.6) and q>=580]
  if candidate:
   n,xl,xr=max(candidate)
   lines.append((y,xl,xr,n))
 detections=[]
 for y,xl,xr,n in lines:
  eligible=[]
  for y2,x2,r2,n2 in lines:
   delta=y2-y
   if not round(ROW_PITCH*.82)<=delta<=round(ROW_PITCH*1.18):continue
   if abs(xr-r2)>round(ROW_PITCH*.31):continue
   # Physically choose the true left edge near the MORE complete horizontal
   # stroke. Soft colour can extend the OTHER stroke into a red scene.
   anchor=max(xl,x2)
   pad=round(ROW_PITCH*.85)
   lo=max(0,anchor-pad);hi=min(W-7,anchor+pad)
   if hi<=lo:continue
   side=hard[y+5:y2-4,lo:hi+1]
   if side.shape[0]<round(ROW_PITCH*.24):continue
   support=np.mean(side,axis=0)
   strongest=float(np.max(support))
   if strongest<.5:continue
   # Tie-break among similarly supported columns by physical proximity to
   # the stronger horizontal edge, not an arbitrary clip/video constant.
   tie=np.flatnonzero(support>=strongest-.04)
   xx=int(lo+min(tie,key=lambda k:abs((lo+int(k))-anchor)))
   rr=min(W-1,max(xr,r2)+3)
   left=max(0,xx-3)
   top=max(0,y-2);bottom=min(H,y2+3)
   if rr-left<round((bottom-top)*3.35) or bottom-top<round(ROW_PITCH*.6):continue
   ix=max(0,xx+6);ir=min(W,rr-6)
   if ir-ix<round(ROW_PITCH*2.5):continue
   if red[min(H-1,y+round(ROW_PITCH*.39)),ix:ir].mean()>.40:continue
   if red[y+delta//2,ix:ir].mean()>.45:continue
   interior_y=red[top+4:bottom-4,:]
   if interior_y.shape[0]<round(ROW_PITCH*.24):continue
   side_band=max(4,round((bottom-top)*.27))
   ls=np.mean(np.any(interior_y[:,left:min(W,left+side_band)],axis=1))
   rs=np.mean(np.any(interior_y[:,max(0,rr-side_band+1):rr+1],axis=1))
   if min(ls,rs)<.64:continue
   # The right side must belong to the same red physical border, not a scene
   # line that happens to end at the same screen x coordinate.
   rhard=hard[y+round(ROW_PITCH*.15):y2-round(ROW_PITCH*.12),max(0,min(xr,r2)-round(ROW_PITCH*.36)):min(W,max(xr,r2)+round(ROW_PITCH*.12))]
   if not rhard.size or np.mean(np.any(rhard,axis=1))<.46:continue
   eligible.append((abs(delta-ROW_PITCH),abs(xx-anchor),y2,left,rr,n2))
  if not eligible:continue
  _,_,y2,left,right,n2=min(eligible)
  top=max(0,y-2);bottom=min(H,y2+3)
  roi=img[top:bottom,left:right+1].copy()
  detections.append(dict(x=int(left+X0),y=int(top+Y0),w=int(right-left+1),h=int(bottom-top),red=int(n+n2),roi=roi))
 filtered=[]
 for r in sorted(detections,key=lambda a:(a['y'],-a['red'])):
  if all(abs(q['y']-r['y'])>round(ROW_PITCH*.46) for q in filtered):filtered.append(r)
 return filtered



class RedCalibration:
    """Unsupervised video-local red-frame calibration from verified physical HUD boxes.

    No title, OCR, expected count, or user-specific calibration. Hue proximity
    and adaptive chroma/brightness are computed in canonical 1080p ROI. The
    conservative four-edge geometry is still required for every new detection.
    """
    def __init__(self):
        self.saturation=[]
        self.value=[]
        self.samples=0

    def masks(self, img):
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        Hh,S,V=(a.astype(np.int16) for a in cv2.split(hsv))
        dh=np.minimum(Hh,180-Hh)
        # Hue wraps around 0/179 in OpenCV; saturation is relative to the
        # independently verified red frame measurements from THIS video.
        s_ref=float(np.median(self.saturation)) if self.saturation else 105.
        v_ref=float(np.median(self.value)) if self.value else 175.
        soft=(dh<=14)&(S>=max(24.,s_ref*.43))&(V>=max(32.,v_ref*.48))
        hard=(dh<=12)&(S>=max(43.,s_ref*.65))&(V>=max(42.,v_ref*.56))
        # Closing fills narrow CRF gaps, never creates arbitrary new outlines;
        # vertical support and dark interior remain independent hard gates.
        k=max(3,int(round(ROW_PITCH*.12))|1)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,1))
        soft=cv2.morphologyEx(soft.astype('uint8'),cv2.MORPH_CLOSE,kernel)>0
        hard=cv2.morphologyEx(hard.astype('uint8'),cv2.MORPH_CLOSE,kernel)>0
        return soft,hard

    def update(self,img,rows):
        if not rows:return
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        for row in rows:
            x=max(0,row['x']-X0);y=max(0,row['y']-Y0)
            h=row['h'];w=row['w'];x1=min(W,x+w);y1=min(H,y+h)
            if h<round(ROW_PITCH*.7) or w<round(h*3.35):continue
            if x1<=x or y1<=y:continue
            crop=hsv[y:y1,x:x1]
            # Only outer HUD border candidate pixels: never learn from scene
            # interiors, weapon whites, text, or unrelated colour objects.
            yy,xx=np.indices(crop.shape[:2]);pad=max(2,round(h*.10))
            outer=(yy<pad)|(yy>=crop.shape[0]-pad)|(xx<pad)|(xx>=crop.shape[1]-pad)
            HH,SS,VV=cv2.split(crop)
            dh=np.minimum(HH.astype('int16'),180-HH.astype('int16'))
            target=outer&(dh<=12)&(SS>35)&(VV>45)
            if target.sum()<round(h*.75):continue
            self.saturation.append(float(np.quantile(SS[target],.25)))
            self.value.append(float(np.quantile(VV[target],.25)))
            self.saturation=self.saturation[-30:];self.value=self.value[-30:]
            self.samples+=1


def detect(img, color_model=None):
    """Preserve strict red border rectangles; add only independent complete
    rectangles from the weaker red/pink hypothesis, with four-edge + dark
    interior verification applied to BOTH masks. Never trust soft alone
    if it conflicts with a strong box in the same row.
    """
    saturated=strict_bmask(img)
    fixed=_detect_with_mask(img,saturated,saturated)
    supplementary=_detect_with_mask(img,bmask(img),saturated)
    hsv_hard=None
    if color_model is not None:
        soft,hard=color_model.masks(img)
        hsv_hard=hard
        # Hard HSV support is calibrated to each video, but strict red is
        # retained, giving a secondary route for low-saturation faded HUDs.
        # HSV hue guides weak *horizontal* strokes only; an independent
        # saturated-red vertical border must still prove a new rectangle.
        supplementary.extend(_detect_with_mask(img,soft,saturated))
    # Try dim-red full rectangles only where a strict/supplementary box is
    # absent at the same row. This never weakens the four-side requirement.
    # Only when the original hard-red detection has no full rectangle:
    # do not let dim-red scene geometry compete with a trusted red HUD row.
    # This fallback is FPS-independent and uses the same four-edge validator.
    if not fixed:
        dim=dim_red_frame_mask(img)
        for row in _detect_with_mask(img,dim,dim):
            if not any(abs(row['y']-old['y'])<=round(ROW_PITCH*.46) for old in supplementary):
                fixed.append(row)
    for row in supplementary:
        if any(abs(row['y']-prev['y'])<=round(ROW_PITCH*.46) for prev in fixed):
            continue
        # A pink/red top edge may be desaturated by the game image.
        # But a TRUE own red rectangle still has a saturated-red lower
        # stroke plus saturated-red vertical side. Non-own amber HUD boxes
        # are rejected even if weak-red geometry alone looks rectangular.
        x=row['x']-X0;y=row['y']-Y0;w=row['w'];h=row['h']
        strong=saturated[max(0,y):min(H,y+h),max(0,x):min(W,x+w)]
        if strong.shape[0]<10 or strong.shape[1]<round(ROW_PITCH*3):
            continue
        inner=strong[:,max(3,round(h*.13)):max(4,strong.shape[1]-round(h*.13))]
        bottom=float(np.max(np.mean(inner[-min(6,h):,:],axis=1)))
        band=max(3,round(h*.24))
        lside=float(np.mean(np.any(strong[4:-4,:band],axis=1)))
        rside=float(np.mean(np.any(strong[4:-4,-band:],axis=1)))
        if bottom<.25 or min(lside,rside)<.25:
            continue
        fixed.append(row)
    result=sorted(fixed,key=lambda x:(x['y'],x['x']))
    if color_model is not None and result:
        # Conservative learning: accept only fully validated four-edge rows.
        color_model.update(img,result)
    return result
