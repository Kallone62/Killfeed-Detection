#!/usr/bin/env python3
"""DescripCS2 candidate: physical CS2 HUD event tracker, source-pixel only.

Reads a 10fps FFmpeg HUD stream and forms four-sided red rectangles. Associates
observations via a Hungarian global assignment, shifted-row pitch, visual
fingerprints, LOST tracks and hysteresis. Refines onset with original 60fps
source frames. OCR and title / expected kill count are never consulted.

The 10fps sources and reports are compatible with the existing JS crop/overlay
stages after the explicit JS interface patch in this experimental package.
NOT production-accepted until full video/identity/frame/PNG regressions pass.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import os
import cv2
import numpy as np

# Bound CPU usage for unattended multi-video batches; no internal OpenCV pool.
cv2.setNumThreads(1)

sys.path.insert(0,str(Path(__file__).resolve().parent))
from hud_detector import source_frames,detect,ROW_PITCH,RedCalibration
from hud_signature import signature,dice
from photometric_v11 import normalized_signature
from d3_fullanchor import fullanchor_fade_onset
from hud_events import build_events
from weak_birth_60 import rescue_weak_events
from hud_ownership import assess_ownership

SCAN_FPS=10
REFINE_FPS=60


def scan(video, start, end, ffmpeg):
    """Complete red-HUD observations; 210px/WIDTH state is NOT an event rule."""
    observations=[]
    red_model=RedCalibration()
    for t,frame in source_frames(video,SCAN_FPS,start=start,end=end,ffmpeg=ffmpeg):
        for rect in detect(frame,red_model):
            glyph=signature(rect['roi'])
            if glyph is None:continue
            rect['sig']=glyph
            rect['t']=t
            observations.append(rect)
    return observations


def stable_reference(event):
    """Clean exemplar, not the possibly still-animating initial HUD frame."""
    all_obs=[o for group in event['source_groups'] for o in group]
    all_obs.sort(key=lambda x:x['t'])
    first=all_obs[0]
    early=[o for o in all_obs if first['t']+.12<=o['t']<=first['t']+.7]
    if not early:early=all_obs
    # Only independently observed native rectangles qualify.
    ref=event['ref']
    def quality(o):
        return (dice(o['sig'],ref,5),o['red']/max(1,o['w']),-abs(o['t']-first['t']-.3))
    return max(early,key=quality)


def anchored_weak_rectangles(img, prior, red_model):
    """Bounded soft-edge evidence for the FIRST frame of an ALREADY confirmed kill.

    Never creates a new event, never overrides strict OWN ownership. At low
    resolution the red rectangle may be visible in HSV but its hard left edge
    is too faint to rediscover by horizontal scanning. We have the independent
    10 Hz track's physical rectangle, so test THAT precise rectangle, including
    quantized HUD movement, for four-edge continuity and coloured victim ink.
    """
    if img is None or img.ndim!=3:
        return []
    soft,_hard=red_model.masks(img)
    ih,iw=img.shape[:2]
    h,w=int(prior['h']),int(prior['w'])
    x=int(prior['x']-1320);y=int(prior['y']-42)
    if h<round(ROW_PITCH*.65) or w<round(h*3):return []
    wiggle=max(1,round(h*.09))
    # No arbitrary position search: only the track's physical edge and legal
    # integer pitch shifts; ±0.09 row height permits blur/registration jitter.
    offsets=(-wiggle,0,wiggle)
    candidates=[]
    for k in (0,):  # D5 exploratory: soft onset cannot borrow an older HUD stack row
        for dy in offsets:
            yy=y+k*ROW_PITCH+dy
            if yy<0 or yy+h>ih:continue
            for dx in offsets:
                xx=x+dx
                if xx<0 or xx+w>iw:continue
                region=soft[yy:yy+h,xx:xx+w]
                edge=max(2,round(h*.11))
                pad=max(3,round(h*.18))
                top=region[:edge,pad:w-pad]
                bottom=region[-edge:,pad:w-pad]
                left=region[edge:h-edge,:edge]
                right=region[edge:h-edge,-edge:]
                if any(z.size==0 for z in (top,bottom,left,right)):continue
                # Each side must be physically continuous, not one red pixel.
                coverage=(float(np.mean(np.any(top,axis=0))),
                          float(np.mean(np.any(bottom,axis=0))),
                          float(np.mean(np.any(left,axis=1))),
                          float(np.mean(np.any(right,axis=1))))
                if min(coverage[0],coverage[1])<.70 or min(coverage[2],coverage[3])<.50:
                    continue
                interior=region[edge:h-edge,pad:w-pad]
                if interior.size==0 or float(interior.mean())>.45:
                    continue
                # The reconstructed rectangle contains ORIGINAL source pixels.
                candidates.append(dict(x=xx+1320,y=yy+42,w=w,h=h,
                                      roi=img[yy:yy+h,xx:xx+w].copy(),
                                      edge_quality=min(coverage)))
    return candidates


def refine_onset(video,event,reference,start,ffmpeg,competitors=()):
    """Refine first source red rectangle at 60 Hz, including bounded visual shifts.

    Tight 5-pixel matching preserves the original onset rule. A wider 8-pixel
    spatial registration is admitted only when a source-matching row has
    demonstrably better victim evidence than ALL alternative event exemplars.
    Neither match can create or remove a 10 Hz EVENT.
    """
    first=event['first']
    lo=max(start,first-.35)
    hi=first+.125
    all_obs=[o for group in event['source_groups'] for o in group]
    first_obs=min(all_obs,key=lambda o:o['t'])
    # D3: only after V11/C already confirmed a physical kill and stable PNG.
    # D2's primary detector and 0.3s native backsearch are left untouched.
    def d3_onset_extension(base_time,base_status,base_score):
        lo3=max(start,min(first-.85,float(base_time)-.85))
        hi3=max(first+.125,float(base_time)+.125)
        if hi3<=lo3:return base_time,base_status,base_score
        # A separate lookback rediscovers the first stable event-specific shape
        # when coarse scan / native strict rectangle was badly delayed.
        # Actual fade search is LIMITED to <=.30s from that full anchor.
        extra=list(source_frames(video,REFINE_FPS,start=lo3,end=hi3,ffmpeg=ffmpeg))
        pred=fullanchor_fade_onset(extra,base_time,reference,reference,competitors)
        if pred is not None and pred[0]<base_time-1/REFINE_FPS+.003:
            return pred[0],'RED_60FPS_HIGHPASS_FULL_ANCHOR_FADE',pred[1]
        return base_time,base_status,base_score
    ref=reference['sig']
    found=[]
    weak_found=[]
    candidate_frames=[]
    red_model=RedCalibration()
    for t,img in source_frames(video,REFINE_FPS,start=lo,end=hi,ffmpeg=ffmpeg):
        candidate_frames.append((t,img))
        for row in detect(img,red_model):
            other=signature(row['roi'])
            if other is None:continue
            edge_delta=abs((row['x']+row['w'])-(first_obs['x']+first_obs['w']))
            pitch=round((row['h']+first_obs['h'])*.5)
            y_shift=min(abs(row['y']-first_obs['y']-k*pitch) for k in (-2,-1,0,1,2))
            if edge_delta>max(8,round(pitch*.75)) or y_shift>max(12,round(pitch*.36)):
                continue
            strict_score=dice(other,ref,5)
            score=strict_score
            if strict_score<.78:
                wide_score=dice(other,ref,8)
                if wide_score<.84:continue
                other_best=max((dice(other,c['sig'],8) for c in competitors if c['sig'] is not None),default=0.)
                if wide_score-other_best < .12:continue
                score=wide_score
            found.append((t,score,row))
        # A second, strictly *onset-only* path: do not promote soft masks to
        # event creation. Require the already verified 10 Hz event's same
        # physical width/right edge, four continuous HSV sides, stable coloured
        # glifs, and agreement for >=3 consecutive 60 Hz source samples.
        good=[]
        for row in anchored_weak_rectangles(img,first_obs,red_model):
            glyph=signature(row['roi'])
            if glyph is None:continue
            score=dice(glyph,ref,8)
            adaptive=max(.65,min(.75,float(event['stability'])-.13))
            if score<adaptive:continue
            competitor=max((dice(glyph,info['sig'],8) for info in competitors
                            if info['first']-.10<=t<=info['last']+.10 and
                            abs(info['width']-row['w'])<=row['h']*.30 and
                            abs(info['right']-(row['x']+row['w']))<=row['h']*.30),
                           default=0.)
            if score-competitor<.12:continue
            good.append((score,row))
        if good:
            score,row=max(good,key=lambda z:z[0]);weak_found.append((t,score,row))
    # Reject one-frame soft-line flashes. The same track-anchored rectangle
    # must recur in at least three consecutive independent 60 Hz observations.
    weak_found.sort(key=lambda x:x[0])
    confirmed=[]
    for i in range(len(weak_found)-2):
        if (weak_found[i+2][0]-weak_found[i][0]<=2/REFINE_FPS+.004 and
                weak_found[i+2][0]-weak_found[i][0]>=2/REFINE_FPS-.004):
            confirmed.append(weak_found[i]);break
    if confirmed and (not found or confirmed[0][0]+.008<min(f[0] for f in found)):
        t,score,rect=confirmed[0]
        physical=t;status='RED_60FPS_ANCHORED_SOFT'
    elif found:
        found.sort(key=lambda x:x[0]);physical,score,_=found[0]
        status='RED_60FPS_CANDIDATE'
    else:
        physical=first;status='RED_60FPS_UNRESOLVED';score=0.
    # Verified EVENT exists already; search at most 0.3s BACKWARDS within
    # the same physical row for a two-frame corroborated colour glyph.
    # No new EVENT and no changes to native victim identity or stable PNG.
    reference_norm=normalized_signature(reference['roi'],signature)
    competitors_norm=[(c['first'],c['last'],normalized_signature(c['roi'],signature)) for c in competitors if c.get('roi') is not None]
    if reference_norm is not None:
        first_obs_right=first_obs['x']+first_obs['w']-1320
        original_y=first_obs['y']-42
        h=int(first_obs['h']);w=int(first_obs['w'])
        candidates=[]
        for t,img in candidate_frames:
            if not physical-.30-1e-5<=t<=physical+.034:continue
            # Fast centre-first geometry search: normalise at most three
            # independently plausible HUD stack positions. Pixel jitter is
            # examined ONLY near a plausible glyph, not in scene background.
            best=None;best_shift=None
            for row_shift in (-1,0,1):
                yy=original_y+row_shift*ROW_PITCH
                xx=first_obs_right-w
                if yy<0 or yy+h>img.shape[0] or xx<0 or xx+w>img.shape[1]:continue
                crop=img[yy:yy+h,xx:xx+w]
                cand=normalized_signature(crop,signature)
                if cand is None:continue
                support=dice(cand,reference_norm,8)
                if best is None or support>best:
                    best=support;best_shift=row_shift
            if best is None or best<.60:continue
            shifts=[(best_shift,0,0)]
            if best<.77:
                shifts += [(best_shift,dy,dx) for dy in (-3,0,3)
                           for dx in (-3,0,3) if dy or dx]
            winning=None
            for row_shift,dy,dx in shifts:
                yy=original_y+row_shift*ROW_PITCH+dy
                xx=first_obs_right-w+dx
                if yy<0 or yy+h>img.shape[0] or xx<0 or xx+w>img.shape[1]:continue
                crop=img[yy:yy+h,xx:xx+w]
                candidate=normalized_signature(crop,signature)
                if candidate is None:continue
                support=dice(candidate,reference_norm,8)
                if support<.77:continue
                # Reject competitors occupying the same physical HUD region.
                rival=max((dice(candidate,other,8) for f,l,other in competitors_norm
                            if other is not None and f-.12<=t<=l+.12),default=0.)
                if support-rival<.12:continue
                if winning is None or support>winning:winning=support
            best=winning
            if best is not None:candidates.append((t,best))
        candidates.sort()
        for ix in range(len(candidates)-2):
            t,confidence=candidates[ix]
            if t>=physical-1e-5:continue
            tn,cn=candidates[ix+1]
            t2,c2=candidates[ix+2]
            if (abs(tn-t-1/REFINE_FPS)<=.004 and
                    abs(t2-t-2/REFINE_FPS)<=.004):
                # Three 60Hz samples span >=2 original 30fps frames, avoiding
                # double-counting a duplicated 30fps image as independent.
                return d3_onset_extension(round(t*REFINE_FPS)/REFINE_FPS,'RED_60FPS_GLYPH_BACKTRACK',min(confidence,cn,c2))
    change=confirmed_gradient_fade_onset(candidate_frames,physical,first_obs,reference,competitors)
    if change is not None:
        gt_time,gt_score=change
        if gt_time+1/REFINE_FPS<physical:
            return d3_onset_extension(gt_time,'RED_60FPS_GRADIENT_FADE_BACKTRACK',gt_score)
    return d3_onset_extension(round(physical*REFINE_FPS)/REFINE_FPS,status,score)


def _grayscale_gradient(img):
    """Source-only photometric correlation insensitive to multiplicative fade gain."""
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return cv2.magnitude(gx,gy)


def _gradient_similarity(a,b):
    if a.shape!=b.shape:return -1.
    aa=a-a.mean();bb=b-b.mean()
    denom=float(np.linalg.norm(aa)*np.linalg.norm(bb))
    return float(np.sum(aa*bb)/denom) if denom>1e-6 else -1.


def confirmed_gradient_fade_onset(candidate_frames,physical,first_obs,reference,competitors):
    """Event-confirmed ONLY: faint first alpha via new-row gradient change point.

    A growing, event-specific *glyph+border* template must emerge in the
    SAME physical slot across three distinct 60Hz samples. Source background
    and older rows are rejected by a pre-fade baseline, an abrupt rise and
    contemporary competing-row matching. This can never create an event.
    """
    h,w=int(first_obs['h']),int(first_obs['w'])
    x,y=int(first_obs['x']-1320),int(first_obs['y']-42)
    ref=reference.get('roi')
    if ref is None or min(h,w)<18 or h<20 or w<round(h*3):return None
    rg=_grayscale_gradient(cv2.resize(ref,(w,h),interpolation=cv2.INTER_LINEAR))
    data=[]
    for t,img in candidate_frames:
        if not physical-.30-1e-5<=t<=physical+.0334:continue
        best=(-1.,None)
        # Precisely track the independently identified physical row: no
        # free scanning across the HUD and no candidate event creation.
        for dy in (-2,0,2):
            for dx in (-2,0,2):
                xx=x+dx;yy=y+dy
                if xx<0 or yy<0 or xx+w>img.shape[1] or yy+h>img.shape[0]:continue
                patch=img[yy:yy+h,xx:xx+w]
                corr=_gradient_similarity(_grayscale_gradient(patch),rg)
                if corr>best[0]:best=(corr,patch)
        if best[1] is not None:data.append((t,best[0],best[1]))
    if len(data)<7:return None
    baseline=[score for t,score,_ in data if t<=physical-.18]
    if len(baseline)<3 or float(np.median(baseline))>.15:return None
    for i in range(1,len(data)-2):
        t,now,patch=data[i]
        tp,prior,_=data[i-1]
        t1,after,_=data[i+1]
        t2,after2,_=data[i+2]
        if t>physical-1/REFINE_FPS+.004:break
        if abs(t-tp-1/REFINE_FPS)>.004 or abs(t1-t-1/REFINE_FPS)>.004 or abs(t2-t1-1/REFINE_FPS)>.004:continue
        if prior>.27 or now<.33 or now-prior<.15 or after<.33 or after2<.52:continue
        # Do not confuse a co-visible old row with this event. Apply the
        # same width, right edge and time overlap semantics as native V11.
        rival=-1.
        for c in competitors:
            croi=c.get('roi')
            if croi is None or not (c['first']-.12<=t<=c['last']+.12):continue
            if abs(c['width']-w)>h*.30 or abs(c['right']-first_obs['x']-w)>h*.30:continue
            crg=_grayscale_gradient(cv2.resize(croi,(w,h),interpolation=cv2.INTER_LINEAR))
            rival=max(rival,_gradient_similarity(_grayscale_gradient(patch),crg))
        if rival>=0 and now-rival<.12:continue
        return round(t*REFINE_FPS)/REFINE_FPS,now
    return None


def _crop_render_row_once(img):
    """Trim a neighbour's red horizontal stroke, never alter event identity.

    When two rectangles touch, the detector may return the previous row's
    bottom stroke a few pixels above the selected row's real top stroke.
    That leftover line is cosmetic leakage into the persistent PNG. Retain
    only the lower complete top border when two independent red lines appear.
    Returns (cropped_source_pixels,removed_top_px). Fail-safe keeps input.
    """
    if img is None or img.ndim!=3:return img,0
    h,w=img.shape[:2]
    if h<22 or w<round(h*3):return img,0
    b,g,r=(ch.astype(np.int16) for ch in cv2.split(img))
    red=(r>125)&(r>g*1.85)&(r>b*1.5)&(g<115)
    interior=red[:,max(3,round(w*.03)):max(4,w-round(w*.03))]
    projection=interior.mean(axis=1)
    hits=np.flatnonzero(projection>.67)
    if not len(hits):return img,0
    runs=[];a=pr=int(hits[0])
    for j in hits[1:]:
        j=int(j)
        if j-pr>1:runs.append((a,pr));a=j
        pr=j
    runs.append((a,pr))
    head=[z for z in runs if z[0] <=round(h*.28)]
    tail=[z for z in runs if z[0] >=round(h*.70)]
    if len(head)<2 or not tail:return img,0
    first,second=head[0],head[1]
    if second[0]-first[1]<2 or second[0]-first[0]>round(h*.28):return img,0
    # Two distinct high-coverage horizontal strokes separated by an empty
    # band cannot both be the top of one normal thin CS2 border. The first
    # is spill from the neighbouring row; keep the lower, complete rectangle.
    trim=max(0,second[0]-2)
    if trim<=0 or h-trim<round(ROW_PITCH*.85):return img,0
    return img[trim:,:,:].copy(),trim


def crop_render_row(img):
    """Repeat ONLY after a complete second upper stroke survives one crop.

    Multiple neighbouring HUD borders can be present above the physically
    confirmed row. Each pass retains a distinct lower four-edge rectangle;
    never resamples, invents ink, or changes the source track identity.
    """
    original=img;total=0
    for _ in range(3):
        candidate,trim=_crop_render_row_once(original)
        if trim<=0:break
        # A two-pixel repeat pass can mistake normal adjacent anti-aliased
        # border rows for separate HUD rectangles. Require a real detached
        # neighbour (>=4 px) for SECOND and later cosmetic trims; the first
        # original crop remains unchanged. Never accept by source/video ID.
        if total>0 and trim<4:break
        total+=trim;original=candidate
    return original,total


def assert_own_track_png(event, obs, rendered_png, competitors=(), source_trim_px=0):
    """Fail closed when a PNG anchor comes from another track/identity.

    A legitimate reidentified fragment may have a different track ID, but its
    raster must agree with this event's native victim glyph signature. The
    observation must be owned by this event's source groups, not sourced
    from a global frame or a different event's representative crop.
    """
    if not any(o is obs for group in event['source_groups'] for o in group):
        raise RuntimeError('PNG_TRACK_PROVENANCE_VIOLATION')
    if obs.get('_track_id') is None or not any(
            o.get('_track_id') == obs['_track_id'] for g in event['source_groups'] for o in g):
        raise RuntimeError('PNG_MISSING_TRACK_ID')
    if obs.get('sig') is None or event.get('ref') is None:
        raise RuntimeError('PNG_MISSING_VICTIM_SIGNATURE')
    own = dice(obs['sig'],event['ref'],5)
    if own < .62:
        raise RuntimeError('PNG_VICTIM_TRACK_MISMATCH native_dice=%.4f' % own)
    raster_sig=signature(rendered_png)
    if raster_sig is None:
        raise RuntimeError('PNG_RENDER_GLYPH_MISMATCH')
    if dice(raster_sig,obs['sig'],8) < .58:
        # Cropping two physically separate, complete red top strokes changes
        # signature-normalisation height, not the rendered victim identity.
        # Permit ONLY a byte-exact, nonempty slice of this exact verified
        # track's ROI, never a different sample or a synthesized glyph.
        exact=(source_trim_px>0 and source_trim_px<obs['roi'].shape[0]
               and np.array_equal(obs['roi'][source_trim_px:,:,:],rendered_png)
               and int(raster_sig.sum())>=12)
        if not exact:
            raise RuntimeError('PNG_RENDER_GLYPH_MISMATCH')
    for other in competitors:
        if other.get('ref') is None:continue
        rival=dice(obs['sig'],other['ref'],5)
        if rival > own + .12:
            raise RuntimeError('PNG_COMPETING_TRACK_IDENTITY rival=%.4f own=%.4f' % (rival,own))
    return {'track_id':obs['_track_id'],'own_dice':own}


def samples_for(event,reference):
    """Select stable, original HUD images for labeling, one anchor per EVENT."""
    obs=sorted((o for g in event['source_groups'] for o in g),key=lambda o:o['t'])
    opts=[reference]
    after=[o for o in obs if o['t']>=reference['t']+.2 and dice(o['sig'],reference['sig'],5)>=.83]
    if after:
        opts.append(after[min(len(after)-1,len(after)//2)])
    else:
        # A heavily compressed fade may destroy the early anchor's native
        # victim colour despite a long, independently confirmed stable track.
        # Only add a SECOND, same-track PNG. Do not change the first anchor,
        # event identity or onset, and never borrow a different track.
        late=[o for o in obs if o is not reference
              and o.get('_track_id')==reference.get('_track_id')
              and reference['t']+.2<=o['t']<=reference['t']+2.5
              and dice(o['sig'],reference['sig'],5)>=.64]
        ranked=[]
        for o in late:
            peers=[q for q in late if q is not o and abs(q['t']-o['t'])<=.45]
            supports=sorted((dice(o['sig'],q['sig'],5) for q in peers),reverse=True)
            if len(supports)<3 or supports[2]<.84:continue
            ranked.append((supports[2],o['red']/max(1,o['w']),-o['t'],o))
        if ranked:
            ranked.sort(key=lambda z:z[:3],reverse=True)
            opts.append(ranked[0][3])
    # Dedupe exact source time/box; the first is the PNG anchor.
    out=[];seen=set()
    for o in opts:
        key=(round(o['t']*10),o['x'],o['y'])
        if key not in seen:seen.add(key);out.append(o)
    return out


def write_report(video,output,asset_dir,events,start,ffmpeg):
    asset_dir=Path(asset_dir)
    asset_dir.mkdir(parents=True,exist_ok=True)
    rows=['# MODE=PHYSICAL_HUD_TRACKING_10FPS_HUNGARIAN_LOST_HYSTERESIS_OWNER_CONSENSUS']
    state_no=0
    uncertain=0
    references=[stable_reference(e) for e in events]
    for event_id,e in enumerate(events,1):
        reference=references[event_id-1]
        others=[dict(sig=r['sig'],width=r['w'],right=r['x']+r['w'],
                     first=e2['first'],last=e2['last'],roi=r['roi'])
                for j,(r,e2) in enumerate(zip(references,events)) if j!=event_id-1]
        onset,status,score=refine_onset(video,e,reference,start,ffmpeg,others)
        if status=='RED_60FPS_UNRESOLVED':uncertain+=1
        state_ids=[]
        for si,obs in enumerate(samples_for(e,reference)):
            state_no+=1
            name='physical_dense_%03d.png'%state_no
            path=asset_dir/name
            render_png,removed_top=crop_render_row(obs['roi'])
            competitors=[ee for j,ee in enumerate(events) if j!=event_id-1]
            try:
                assert_own_track_png(e,obs,render_png,competitors,removed_top)
            except RuntimeError as exc:
                if str(exc)=='PNG_RENDER_GLYPH_MISMATCH' and removed_top>0:
                    # Preserve the real, same-track victim if cosmetic trim
                    # clipped glyph ink. The complete invariant still applies.
                    render_png=obs['roi'].copy();removed_top=0
                    assert_own_track_png(e,obs,render_png,competitors,removed_top)
                elif str(exc).startswith('PNG_COMPETING_TRACK_IDENTITY') and si>0:
                    # The original second-state selection can land on an
                    # ambiguous transient despite stronger SAME-TRACK source
                    # frames. Try only independent 4-edge observations of
                    # this event/track, retaining two verified PNGs.
                    pool=[candidate for g in e['source_groups'] for candidate in g
                          if candidate is not obs and candidate['_track_id']==obs['_track_id']
                          and candidate['t']>=reference['t']+.2
                          and dice(candidate['sig'],reference['sig'],5)>=.83]
                    def own_advantage(candidate):
                        own=dice(candidate['sig'],e['ref'],5)
                        rival=max((dice(candidate['sig'],other['ref'],5)
                                   for other in competitors if other.get('ref') is not None),default=0.)
                        return (own-rival,own,-abs(candidate['t']-obs['t']),-candidate['t'])
                    pool.sort(key=own_advantage,reverse=True)
                    repaired=False
                    for alternative in pool:
                        patch,trim=crop_render_row(alternative['roi'])
                        try:
                            assert_own_track_png(e,alternative,patch,competitors,trim)
                        except RuntimeError as e2:
                            if str(e2)=='PNG_RENDER_GLYPH_MISMATCH' and trim>0:
                                patch=alternative['roi'].copy();trim=0
                                try:assert_own_track_png(e,alternative,patch,competitors,trim)
                                except RuntimeError:continue
                            else:continue
                        obs=alternative;render_png=patch;removed_top=trim;repaired=True
                        break
                    if not repaired:raise
                else:raise
            if not cv2.imwrite(str(path),render_png):
                raise RuntimeError('Could not write original HUD crop '+str(path))
            persisted=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if persisted is None or not np.array_equal(persisted,render_png):
                raise RuntimeError('PNG_PERSISTENCE_MISMATCH '+str(path))
            slot=int(round((obs['y']-73)/max(1,ROW_PITCH)))
            slot=max(0,slot)
            # SOURCE time is ABSOLUTE MP4 source time, not trimmed relative time.
            rows.append('SOURCE\t%d\t%d\t%.6f\t%d\t%d\t%d\t%d\t%s'%(
                state_no,slot,obs['t'],obs['x'],obs['y']+removed_top,obs['w'],render_png.shape[0],path.resolve()))
            rows.append('STATE\t%d\t%s\t%d\t%.5f\t%d\t%d'%(
                state_no,'OWN_NEW' if si==0 else 'OWN_CONTINUATION',event_id,
                1.,obs['red'],int(obs['sig'].sum())))
            state_ids.append(state_no)
        # Keep original V14 filtergraph/trim-relative overlay contract.
        rows.append('EVENT\t%d\t%.6f\t%d\t%.6f\t%s'%(
            event_id,max(0.,onset-start),state_ids[0],max(0.,e['last']-start),
            ','.join(map(str,state_ids))))
        rows.append('# ONSET_QUALITY %d %s %.4f'%(event_id,status,score))
        rows.append('# OWNERSHIP %d %s'%(event_id,e.get('ownership','UNKNOWN')))
    Path(output).write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return uncertain


def run(video,output,asset_dir,ffmpeg,start,end,tesseract="tesseract",owner_name=""):
    if not Path(video).is_file():raise FileNotFoundError(video)
    if end<=start:raise ValueError('Invalid source time window')
    observations=scan(video,start,end,ffmpeg)
    events,tracks,segments=build_events(observations)
    events=rescue_weak_events(video,events,segments,ffmpeg)
    if not events:raise RuntimeError('No source-confirmed own red HUD events')
    # A complete red rectangle is not sufficient evidence of WHO attacked.
    # The spectator/camera can change within one encoded clip. Distinct source
    # attacker names must be excluded even when their physical box is complete.
    candidate_images=[]
    for e in events:
        ref=stable_reference(e)
        candidate_images.append([o['roi'] for o in samples_for(e,ref)])
    owner_labels,evidence=assess_ownership(candidate_images,tesseract=tesseract,owner_name=owner_name)
    included=[];vetoed=[]
    for i,(e,owner) in enumerate(zip(events,owner_labels)):
        e['ownership']=owner
        if owner=='OTHER':vetoed.append(i+1)
        else:included.append(e)
    print('SOURCE_ATTACKER_CONSENSUS owner=%s before=%d excluded=%s unknown=%d' %
          (evidence.get('owner_prefix'),len(events),','.join(map(str,vetoed)) or 'none',sum(x=='UNKNOWN' for x in owner_labels)),flush=True)
    events=included
    if not events:raise RuntimeError('No attacker-consistent own kill events')
    uncertain=write_report(video,output,asset_dir,events,start,ffmpeg)
    print('PHYSICAL_TRACKER SOURCE_ROWS=%d TRACKS=%d EVENTS=%d FRAME_UNRESOLVED=%d'%(
        len(observations),len(tracks),len(events),uncertain),flush=True)
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    parser.add_argument('--start',required=True,type=float)
    parser.add_argument('--end',required=True,type=float)
    parser.add_argument('--source-dir',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--tesseract',default='tesseract')
    parser.add_argument('--owner-name',default=os.environ.get('DCS_OWNER_NICK',''))
    args=parser.parse_args()
    try:sys.exit(run(args.video,args.out,args.source_dir,args.ffmpeg,args.start,args.end,args.tesseract,args.owner_name))
    except Exception as exc:
        print('PHYSICAL_ROW_TRACK_ERROR: '+str(exc),file=sys.stderr)
        sys.exit(2)
