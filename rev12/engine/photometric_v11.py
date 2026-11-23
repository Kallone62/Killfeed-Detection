"""NON-PRODUCTION V11 experiment: photometric support ONLY for existing tracks.

The frozen hud_signature.signature() and original-event detector remain untouched.
Opt-in pre-comparison canonical height, victim-slice LAB contrast/CLAHE, then
call the SAME chromatic-ink binarizer and Dice scorer. Weak observations must
never open or split events; only extend an independently confirmed strict track.
No per-video tuning, no new service or learned model. Requires 40/40 V11 gate.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

CANONICAL_HEIGHT = 36
EXTENSION_MAX_GAP_SECONDS = 0.35
STRONG_NEW_EVENT_MIN_SECONDS = 0.30


def normalize_victim_roi(bgr, height=CANONICAL_HEIGHT):
    """Return aspect-preserving row at canonical height; edit only victim strip.

    Rescale HSV value at the victim slice 98th percentile and apply bounded
    local CLAHE to the same luminance channel. Hue/saturation are preserved
    and the original chromatic-ink glyph-mask formula survives.
    The source BGR pixels and the original signature function are unchanged.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return None
    h, w = bgr.shape[:2]
    if h < 18 or w < 3*h:
        return None
    dst_w=max(1,int(round(w * height / h)))
    r=cv2.resize(bgr,(dst_w,height), interpolation=cv2.INTER_AREA if h>height else cv2.INTER_LINEAR)
    h,w=r.shape[:2]
    padright=max(6,round(h*.15))
    take=min(w-padright,max(95,round(h*4.6)))
    x0=w-padright-take
    x1=w-padright
    if x0<0 or x1-x0<24:
        return None
    section=r[:,x0:x1]
    hsv=cv2.cvtColor(section,cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(hsv)
    high=float(np.quantile(V,.98))
    if high>10.:
        factor=float(np.clip(235. / high,.75,3.5))
        V=np.clip(V.astype(np.float32)*factor,0,255).astype(np.uint8)
    clahe=cv2.createCLAHE(clipLimit=1.25,tileGridSize=(4,2))
    V=clahe.apply(V)
    r[:,x0:x1]=cv2.cvtColor(cv2.merge((H,S,V)),cv2.COLOR_HSV2BGR)
    return r


def normalized_signature(bgr, signature_fn):
    """Use V11 signature representation; only pixel input is normalized."""
    row=normalize_victim_roi(bgr)
    return signature_fn(row) if row is not None else None


@dataclass(frozen=True)
class ConfirmedTrack:
    event_id: int
    time_s: float
    right_x: float
    y: float
    w: float
    h: float
    reference_bgr: np.ndarray
    independent_strong_observations: int


def strong_hysteresis(timestamps, required_seconds=STRONG_NEW_EVENT_MIN_SECONDS):
    """Seconds-based gating; weak observations must not be passed in here."""
    ordered=sorted(set(round(float(t),6) for t in timestamps))
    return len(ordered)>=4 and ordered[-1]-ordered[0]>=required_seconds-.002


def weak_extension(candidate_bgr, at_time_s, x, y, w, h,
                   confirmed_tracks, signature_fn, dice_fn):
    """Return id of an ALREADY CONFIRMED strong event, or None. Never make an event.

    The caller only supplies low-photometric-confidence four-side *candidate*
    boxes to this route; geometry and native strict ownership must already be
    established upstream. Uses seconds, not number of 10fps samples.
    """
    csig=normalized_signature(candidate_bgr,signature_fn)
    if csig is None:return None
    matches=[]
    for tr in confirmed_tracks:
        if tr.event_id<=0 or tr.independent_strong_observations<4:continue
        if abs(float(at_time_s)-tr.time_s)>EXTENSION_MAX_GAP_SECONDS:continue
        pitch=max(1,(h+tr.h)*.5)
        if abs((x+w)-tr.right_x)>pitch*.42:continue
        if abs(w-tr.w)>pitch*.35:continue
        if min(abs(y-tr.y-k*pitch) for k in (-2,-1,0,1,2))>pitch*.34:continue
        ref=normalized_signature(tr.reference_bgr,signature_fn)
        if ref is None:continue
        value=dice_fn(csig,ref,5)
        if value>=.75:matches.append((value,tr.event_id))
    if not matches:return None
    matches.sort(reverse=True)
    if len(matches)>1 and matches[0][0]-matches[1][0]<.12:return None
    return matches[0][1]
