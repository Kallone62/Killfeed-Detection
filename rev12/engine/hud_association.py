"""Source-only experimental association: Hungarian + LOST tracks + hysteresis.
Never reads a video title, expected count, ground truth or OCR to create events.
Input is the independently decoded red-HUD observation stream.
"""
from __future__ import annotations
import collections
import cv2, numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    # Exact small-matrix Hungarian-equivalent minimum-cost assignment for the
    # Windows embedded runtime without SciPy. Not greedy matching.
    from functools import lru_cache
    def linear_sum_assignment(C):
        C=np.asarray(C,float)
        n,m=C.shape
        # CS2 has <=6 simultaneous HUD rows; feasible to solve exactly.
        if n>m:
            cols,rows=linear_sum_assignment(C.T)
            return rows,cols
        if m>16:
            raise ValueError('Too many simultaneous HUD detections')
        @lru_cache(None)
        def solve(i,used):
            if i==n:return (0.,())
            best=(float('inf'),())
            for j in range(m):
                if used>>j&1:continue
                cost,tail=solve(i+1,used|(1<<j))
                candidate=(float(C[i,j])+cost,(j,)+tail)
                if candidate[0]<best[0]:best=candidate
            return best
        _,ix=solve(0,0)
        return np.arange(n,dtype=int),np.array(ix,dtype=int)

from hud_signature import signature as original_signature
from photometric_v11 import normalized_signature

MIN_INK=25
OBSERVATION_HZ=10.0  # invariant scan clock, independent of MP4 source FPS
LOST_SECONDS=4.6
NEW_STABLE_SECONDS=.20

# cv2 TM_CCORR computes all bounded raster offsets in C; its dot-products
# produce aligned binary Dice. Not fixed names, titles or reference outcomes.
def dice_cv(a,b,offset=3):
    if a is None or b is None:return 0.
    a=np.asarray(a,dtype=np.uint8);b=np.asarray(b,dtype=np.uint8)
    mass=int(a.sum())+int(b.sum())
    if not mass:return 0.
    src=np.pad(a,((offset,offset),(offset,offset)))
    raw=cv2.matchTemplate(src.astype('float32'),b.astype('float32'),cv2.TM_CCORR)
    return float(min(1,2*raw.max()/mass))

class Track:
    def __init__(self,i,o,t):
        self.i=i;self.first=t;self.last=t;self.x=o['x'];self.y=o['y'];self.h=o['h'];self.w=o['w']
        o['_track_id']=i
        self.ref=o['sig'];self.samples=[o];self.status='TENTATIVE'
    def append(self,o,t,score):
        o['_track_id']=self.i
        self.last=t;self.x=o['x'];self.y=o['y'];self.h=o['h'];self.w=o['w'];self.samples.append(o)
        self.status='ACTIVE' if t-self.first>=NEW_STABLE_SECONDS else 'TENTATIVE'
        # Keep initial independent visual reference instead of chaining every noisy mask.
        if self.ref is None and o['sig'] is not None:self.ref=o['sig']

def candidate_cost(o,tr,t):
    age=(t-tr.last)
    if age < -1e-6 or age > LOST_SECONDS:return 1e4
    # HUD rows have a known discrete stack pitch. k=0 means no shift; new
    # kills displace the same source row by k*h, where h is observed HUD height.
    row_pitch=round((o['h']+tr.h)*.5)
    steps=list(range(-3,5))
    dy=min(abs(o['y']-tr.y-k*row_pitch) for k in steps)
    k=min(steps,key=lambda k:abs(o['y']-tr.y-k*row_pitch))
    dx=abs((o['x']+o['w'])-(tr.x+tr.w))  # align HUD right edge, not variable left edge
    if dy>row_pitch*.50 or dx>row_pitch*.52:return 1e4
    # D6: existing older CS2 killfeed rows can stay or rise in the stack,
    # not jump DOWN into a newly introduced lower row with similar glyph ink.
    if k>0:return 1e4
    # Normal 3-px match and wider match only when clearly position-compatible.
    similarity=dice_cv(o['sig'],tr.ref,offset=3)
    if similarity<0.7 and (k!=0 or age>.22):
        similarity=max(similarity,dice_cv(o['sig'],tr.ref,offset=8))
    # Only a previously stable physical row may receive photometric fallback.
    # The native chromatic signature remains first and governs all births.
    if (similarity < .70 and tr.status in ('ACTIVE','LOST')
            and len(tr.samples)>=4 and 0 <= age <=.35 and abs(k)<=1
            and dx<=row_pitch*.27 and abs(o['w']-tr.w)<=row_pitch*.30):
        recent=tr.samples[-1]
        a=normalized_signature(o['roi'],original_signature)
        b=normalized_signature(recent['roi'],original_signature)
        ns=dice_cv(a,b,offset=5)
        if ns>=.77 and similarity>=.40:
            similarity=max(similarity,ns)
    if age<=.22 and k==0 and similarity<.70:return 1e4  # require a same-identity visual fit
    if age>.22 and similarity<.85:return 1e4  # LOST re-ID requires a strong glyph match
    if k!=0 and similarity<.75:return 1e4  # stacked-row movement requires raster evidence
    # A similar glyph from a different existing row should not jump into this track.
    # Multiple same-frame detections are one-to-one by Hungarian assignment.
    pos=dy/max(1,row_pitch*.45)+dx/max(1,row_pitch*.76) + .14*abs(k)
    appearance=2.1*(1-similarity)
    gap=.10*age if age<.5 else .20*age
    return pos+appearance+gap

def tracks_for(rows):
    observations=collections.defaultdict(list)
    for idx,o in enumerate(rows):
        if o['sig'] is None or o['sig'].sum()<MIN_INK or o['sig'].sum()>.27*o['sig'].size:continue
        observations[round(o['t']*OBSERVATION_HZ)].append((idx,o))
    tracks=[]
    for frame,obs in sorted(observations.items()):
        t=frame/OBSERVATION_HZ
        for tr in tracks:
            if t-tr.last>.2 and tr.status=='ACTIVE':tr.status='LOST'
        eligible=[tr for tr in tracks if t-tr.last<=LOST_SECONDS]
        if eligible:
            C=np.full((len(eligible),len(obs)),1e4,dtype=float)
            for i,tr in enumerate(eligible):
                for j,(idx,o) in enumerate(obs):C[i,j]=candidate_cost(o,tr,t)
            matches=linear_sum_assignment(C)
            used=set()
            for i,j in zip(*matches):
                if C[i,j]>=2.55:continue
                idx,o=obs[j];tr=eligible[i]
                tr.append(o,t,float(C[i,j]));used.add(j)
        else:used=set()
        for j,(idx,o) in enumerate(obs):
            if j not in used:tracks.append(Track(len(tracks)+1,o,t))
    # stable in multiple independent 100ms source frames
    return [tr for tr in tracks if len({round(o['t']*OBSERVATION_HZ) for o in tr.samples})>=3 and tr.samples[-1]['t']-tr.samples[0]['t']>=NEW_STABLE_SECONDS-1e-3]

