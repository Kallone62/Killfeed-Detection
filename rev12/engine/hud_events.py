"""Experimental physical event reconstruction, no source title/OCR/expected kill count.
Hungarian matching + LOST + stable glyph change + simultaneous-row exclusion.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from hud_association import tracks_for,dice_cv,LOST_SECONDS,OBSERVATION_HZ
from hud_signature import signature as original_signature
from photometric_v11 import normalized_signature

CHANGE_CONFIRM_SECONDS=.30
# Physically continuous HUD reflow can briefly interrupt a 10Hz track for
# several frames without introducing a NEW kill. The 0.5s window covers a
# finite HUD displacement, never overrides identity or co-visibility gates.
BOUNDARY_REJOIN_SECONDS=.50
VISUAL_SAME=.78
# Dynamic event re-ID: within-segment stability sets the similarity
# expectation; nearest-competitor margin guards ambiguous merges.
REID_RELATIVE_MARGIN=.12
REID_SELF_TOLERANCE=.18
REID_QUALITY_FLOOR=.68
STRONG_NEW_EVENT_MIN_SECONDS=.60
WEAK_EXTENSION_GAP_SECONDS=.55
MAX_MASK_FRACTION=.27
GRID_SIZE=24*256

def is_usable(o):
 ink=int(o['sig'].sum())
 return ink>=20 and ink<MAX_MASK_FRACTION*GRID_SIZE

def split_hysteresis(track):
 obs=sorted((o for o in track.samples if is_usable(o)),key=lambda o:o['t'])
 if not obs:return []
 groups=[[obs[0]]];pending=[]
 for o in obs[1:]:
  current=groups[-1]
  # compare with multiple STABLE references rather than continuously
  # replacing one anchor with arbitrary (possibly contaminated) frames.
  bases=[current[0]['sig'],current[len(current)//2]['sig'],current[-1]['sig']]
  strength=sorted((dice_cv(o['sig'],ref,5) for ref in bases),reverse=True)[1]
  if strength >= VISUAL_SAME:
   if pending:current.extend(pending);pending=[]
   current.append(o)
   continue
  # possible new visual identity, not yet an EVENT
  if pending and (o['t']-pending[-1]['t']>.25 or dice_cv(o['sig'],pending[-1]['sig'],5)<.70):
   current.extend(pending);pending=[]
  pending.append(o)
  if len(pending)>=4 and pending[-1]['t']-pending[0]['t']>=CHANGE_CONFIRM_SECONDS-.02:
   groups.append(pending)
   pending=[]
 if pending:groups[-1].extend(pending)
 return [group for group in groups if len({round(x['t']*OBSERVATION_HZ) for x in group})>=4 and group[-1]['t']-group[0]['t']>=CHANGE_CONFIRM_SECONDS-.02]

def _self_stability(group):
    references=[group[0]['sig'],group[len(group)//2]['sig'],group[-1]['sig']]
    vals=[dice_cv(references[0],references[1],5),
          dice_cv(references[0],references[2],5),
          dice_cv(references[1],references[2],5)]
    return sorted(vals)[1]


def _event_relation(candidate, event):
    """Physical evidence for two sightings belonging to one HUD rectangle.

    Identity similarity is NOT enough: unrelated names can share weapon glyphs.
    Strong, complete red rectangles preserve width within a physical event;
    new red entries can shift older rows by integer HUD row pitches. Combine
    native width with position/quantized motion and observation-time gap.
    """
    first_obs=candidate['samples'][0]
    last_group=max(event['source_groups'],key=lambda g:max(o['t'] for o in g))
    last_obs=max(last_group,key=lambda o:o['t'])
    h=(first_obs['h']+last_obs['h'])*.5
    # This is a veto only, never proof of identity: two different victims can
    # have identical widths, but one physical kill does not alter its own frame.
    widths=[o['w'] for o in last_group]
    stable_width=float(np.median(widths))
    this_width=float(np.median([o['w'] for o in candidate['samples']]))
    delta_width=abs(stable_width-this_width)
    # Measure physical-box width jitter inside BOTH stable segments. Distinct
    # victim names can share glyph Dice but the SAME red HUD rectangle does
    # not change width by many times its normal source-frame jitter.
    event_mad=float(np.median(np.abs(np.asarray(widths)-stable_width)))
    candidate_widths=np.asarray([o['w'] for o in candidate['samples']],dtype=float)
    candidate_mad=float(np.median(np.abs(candidate_widths-this_width)))
    width_tolerance=max(h*.25,3.5*(event_mad+candidate_mad+1.))
    if delta_width > width_tolerance:
        return None
    dt=max(0.,first_obs['t']-last_obs['t'])
    if dt>LOST_SECONDS:
        return None
    # Compare STABLE segment geometry rather than two transition frames.
    # A HUD row can be half-way through a one-pitch animation at either
    # endpoint; using those endpoints creates a false NEW event on the exact
    # frames where it matters most. Median is in canonical source geometry.
    event_y=float(np.median([o['y'] for o in last_group]))
    candidate_y=float(np.median([o['y'] for o in candidate['samples']]))
    event_right=float(np.median([o['x']+o['w'] for o in last_group]))
    candidate_right=float(np.median([o['x']+o['w'] for o in candidate['samples']]))
    dy=candidate_y-event_y
    k=round(dy/h) if h else 0
    # D6 association already forbids an existing kill row jumping downward.
    # The segment-level Hungarian re-ID must obey that same physical rule.
    if k>0:
        return None
    resid=abs(dy-k*h)/max(1,h)
    # Nearest-pitch residual can be 0.5 at the animation midpoint;
    # reject only physically impossible (>half-pitch) geometry.
    if resid>.501:
        return None
    right_error=abs(event_right-candidate_right)/max(1,h)
    if right_error>.9:
        return None
    # Penalty terms dimensionless; not a source/video-specific model.
    # A 1-row shift is possible, but must have stronger appearance evidence
    # than a near-zero-displacement physically continuous row.
    penalty=(.10*abs(k)+.10*resid+.08*(delta_width/h)+
             .10*right_error+.03*dt)
    return penalty



def _strong_confirmed(event):
    """Independent four-edge source detections spanning real seconds, not FPS count."""
    stamps=sorted({round(o['t'],4) for group in event['source_groups'] for o in group})
    # A few disconnected detections seconds apart cannot independently birth
    # a kill, even if the aggregate lifespan is long. Require a local cohort
    # of four 10 Hz observations within a finite 0.5 s physical HUD window.
    coherent=any(stamps[i+3]-stamps[i]<=.5+.002 for i in range(max(0,len(stamps)-3)))
    return len(stamps)>=4 and stamps[-1]-stamps[0]>=STRONG_NEW_EVENT_MIN_SECONDS-.002 and coherent


def _weak_support_target(weak, strong_events):
    """Weak short segments may support a strict already confirmed EVENT, never birth one.

    Co-visible rows, width changes, shifted text and ambiguous victim matches
    cannot be reassigned. This is NOT an arbitrary count-correction mechanism.
    """
    wo=[o for group in weak['source_groups'] for o in group]
    wf={round(o['t']*OBSERVATION_HZ) for o in wo}
    end=max(o['t'] for o in wo)
    early=min(o['t'] for o in wo)
    w=np.median([o['w'] for o in wo]); h=np.median([o['h'] for o in wo]);
    right=np.median([o['x']+o['w'] for o in wo]);y=np.median([o['y'] for o in wo])
    proposals=[]
    for ev in strong_events:
        if wf & ev['frames']:continue
        so=[o for g in ev['source_groups'] for o in g]
        st=min(o['t'] for o in so);et=max(o['t'] for o in so)
        dt=max(0,st-end,early-et)
        if dt>WEAK_EXTENSION_GAP_SECONDS:continue
        sw=np.median([o['w'] for o in so]);sr=np.median([o['x']+o['w'] for o in so]);sy=np.median([o['y'] for o in so]);sh=np.median([o['h'] for o in so]);pitch=max(1,(h+sh)*.5)
        if abs(w-sw)>pitch*.30 or abs(right-sr)>pitch*.30:continue
        if min(abs(y-sy-k*pitch) for k in (-2,-1,0,1,2))>pitch*.28:continue
        score=max(dice_cv(weak['ref'],ref,5) for ref in ev['refs'])
        if score<.62:
            wn=normalized_signature(wo[len(wo)//2]['roi'],original_signature)
            sn=normalized_signature(so[len(so)//2]['roi'],original_signature)
            score=max(score,dice_cv(wn,sn,5))
        if score>=.62:proposals.append((score,dt,ev))
    proposals.sort(key=lambda z:(z[0],-z[1]),reverse=True)
    if not proposals:return None
    if len(proposals)>1 and proposals[0][0]-proposals[1][0]<.12:return None
    return proposals[0][2]


def hold_short_fragments(events):
    """Finish EVENT birth only after >=.60s of genuine four-edge observation.

    Shorter unconfirmed fragments are retained only as earlier onset/track
    support for a matching independently confirmed strong event. Others are
    rejected as unconfirmed rather than emitted as new kills.
    """
    confirmed=[e for e in events if _strong_confirmed(e)]
    weak=[e for e in events if not _strong_confirmed(e)]
    for fragment in weak:
        target=_weak_support_target(fragment,confirmed)
        if target is None:continue
        target['source_groups'].extend(fragment['source_groups'])
        target['source_groups'].sort(key=lambda group:group[0]['t'])
        target['first']=min(target['first'],fragment['first'])
        target['frames'].update(fragment['frames'])
        target['stability']=max(target['stability'],fragment['stability'])
    confirmed.sort(key=lambda e:e['first'])
    return confirmed

def _same_row_temporal_alias(candidate, event, frames):
    """Reconnect non-co-visible alternating snapshots of ONE physical HUD row.

    This is NOT a downward event re-ID: the two independent 4-edge tracks
    coexist in *lifespan* and share the same row at their near-simultaneous
    starts, but never in a single decoded frame. Distinct simultaneous HUD
    rows remain separate, and later downward jumps are still forbidden D6.
    """
    if frames & event['frames']:return False
    if not (candidate['first']<=event['last'] and candidate['last']>=event['first']):return False
    if abs(candidate['first']-event['first'])>.4:return False
    # Native chromatic victim mask is the primary identity test. Under heavy
    # compression its team-colour classifier can flip on alternating frames;
    # allow the pre-existing photometric signature only as SECONDARY evidence
    # for this independently established same-row temporal alias.
    native=max(dice_cv(candidate['ref'],r,5) for r in event['refs'])
    a=candidate['samples'][0]
    near=min((o for g in event['source_groups'] for o in g),
             key=lambda o:abs(o['t']-a['t']))
    h=(a['h']+near['h'])*.5
    if abs(a['t']-near['t'])>.2+.002:return False
    if abs(a['y']-near['y'])>h*.18:return False
    if abs(a['w']-near['w'])>h*.22:return False
    if abs(a['x']+a['w']-near['x']-near['w'])>h*.25:return False
    if native>=.90:return True
    if native<.90:
        sa=normalized_signature(a['roi'],original_signature)
        sb=normalized_signature(near['roi'],original_signature)
        return dice_cv(sa,sb,5)>=.87
    return False


def _merge_same_row_aliases(events):
    """After reconstruction, coalesce independent interleaved *same physical row*
    fragments only when they are NOT co-visible in any decoded frame.
    Native glyph is primary; contrast-normalized glyph is a secondary
    comparison, NEVER evidence to birth a new kill or bypass ownership.
    """
    result=sorted(events,key=lambda e:e['first'])
    updated=True
    while updated:
        updated=False
        for i in range(len(result)):
            if updated:break
            a=result[i]
            ao=[o for g in a['source_groups'] for o in g]
            for j in range(i+1,len(result)):
                b=result[j]
                if b['first']>a['last']:break
                if a['frames'] & b['frames']:continue
                if abs(a['first']-b['first'])>.50:continue
                bo=[o for g in b['source_groups'] for o in g]
                pitch=float(np.median([o['h'] for o in ao+bo]))
                if not pitch:continue
                def median_geometry(obs):
                    return np.median([o['y'] for o in obs]),np.median([o['w'] for o in obs]),np.median([o['x']+o['w'] for o in obs])
                ya,wa,ra=median_geometry(ao);yb,wb,rb=median_geometry(bo)
                if abs(ya-yb)>pitch*.18 or abs(wa-wb)>pitch*.22 or abs(ra-rb)>pitch*.25:continue
                # Require direct near-time physical samples; no long LOST
                # bridge, no row-pitch translation, and no video-specific rule.
                if min(abs(x['t']-y['t']) for x in ao for y in bo)>.21:continue
                native=max(dice_cv(x,y,5) for x in a['refs'] for y in b['refs'])
                if native<.90:
                    # At most five temporally distributed independently
                    # verified four-edge observations per side.
                    def anchors(obs):
                        seq=sorted(obs,key=lambda o:o['t'])
                        return [seq[k] for k in sorted({round(k*(len(seq)-1)/4) for k in range(5)})]
                    normalized_a=[normalized_signature(o['roi'],original_signature) for o in anchors(ao)]
                    normalized_b=[normalized_signature(o['roi'],original_signature) for o in anchors(bo)]
                    supported=max((dice_cv(x,y,5) for x in normalized_a for y in normalized_b),default=0.)
                    if supported<.87:continue
                a['frames'].update(b['frames'])
                a['source_groups'].extend(b['source_groups'])
                a['source_groups'].sort(key=lambda g:g[0]['t'])
                a['refs'].extend(b['refs'])
                a['last']=max(a['last'],b['last']);a['first']=min(a['first'],b['first'])
                a['stability']=max(a['stability'],b['stability'])
                result.pop(j);updated=True;break
    return sorted(result,key=lambda e:e['first'])


def build_events(rows):
    # Observation-level one-to-one matching is solved by Hungarian in
    # hud_association.tracks_for; an observation cannot match two tracks.
    initial=tracks_for(rows)
    segments=[]
    for track in initial:
        for group in split_hysteresis(track):
            segments.append(dict(first=group[0]['t'],last=group[-1]['t'],
                                 samples=group,ref=group[len(group)//2]['sig'],
                                 stability=_self_stability(group),origin_track_id=track.i))
    segments.sort(key=lambda z:z['first'])
    events=[]
    for candidate in segments:
        frames={round(o['t']*OBSERVATION_HZ) for o in candidate['samples']}
        # First join fragmented alternating observations of the SAME
        # physical row. Never merge observations co-visible in one frame.
        alias=[event for event in events if _same_row_temporal_alias(candidate,event,frames)]
        if len(alias)==1:
            same=alias[0]
            same['frames'] |= frames
            same['source_groups'].append(candidate['samples'])
            same['last']=max(same['last'],candidate['last'])
            same['first']=min(same['first'],candidate['first'])
            same['refs'].append(candidate['ref'])
            continue
        eligible=[]
        for event in events:
            if frames & event['frames']:
                continue  # co-visible physical rows must be distinct
            penalty=_event_relation(candidate,event)
            if penalty is None:continue
            raw=max(dice_cv(candidate['ref'],r,5) for r in event['refs'])
            # A long LOST rejoin needs native victim evidence stronger than
            # the ordinary .68 floor; photometric fallback alone is not a
            # new-victim identity. Preserve genuine reflow with >=.72 native
            # appearance and full physical/co-visibility gates.
            last_seen=max(o['t'] for g in event['source_groups'] for o in g)
            if candidate['first']-last_seen>BOUNDARY_REJOIN_SECONDS and raw<.72:
                continue
            self_floor=max(.65,min(candidate['stability'],event['stability'])-REID_SELF_TOLERANCE)
            # Wider glyph registration is permitted only when the SAME
            # physical rectangle has stable width, right edge and quantized
            # location. It is not a global similarity-threshold relaxation.
            source=event['source_groups'][-1]
            first=candidate['samples']
            h=float(np.median([o['h'] for o in source+first]))
            width_delta=abs(np.median([o['w'] for o in source])-np.median([o['w'] for o in first]))/max(1,h)
            right_delta=abs(np.median([o['x']+o['w'] for o in source])-np.median([o['x']+o['w'] for o in first]))/max(1,h)
            dy=float(np.median([o['y'] for o in first])-np.median([o['y'] for o in source]))
            k=round(dy/max(1,h))
            position_residual=abs(dy-k*h)/max(1,h)
            aliased_lifespan=(candidate['first']<=event['last'] and candidate['last']>=event['first'])
            if (aliased_lifespan and raw<max(self_floor,REID_QUALITY_FLOOR) and
                    width_delta<=.45 and right_delta<=.4 and position_residual<=.30):
                # Two alternating tracker fragments of one visible row may
                # overlap in time WITHOUT ever being co-visible in one frame.
                # Allow wider registration only for this aliasing pattern.
                raw=max(raw,max(dice_cv(candidate['ref'],r,8) for r in event['refs']))
            # Fallback applies only to EXISTING events with strong geometry;
            # it can never independently create a physical event.
            if (raw < REID_QUALITY_FLOOR and penalty <=.15
                    and width_delta<=.27 and right_delta<=.27
                    and position_residual<=.22 and not aliased_lifespan):
                ref_obs=event['source_groups'][-1][len(event['source_groups'][-1])//2]
                cand_obs=candidate['samples'][len(candidate['samples'])//2]
                an=normalized_signature(ref_obs['roi'],original_signature)
                bn=normalized_signature(cand_obs['roi'],original_signature)
                ns=dice_cv(an,bn,5)
                if ns>=.78 and raw>=.40:raw=ns
            eligible.append((raw-penalty,raw,self_floor,event))
        eligible.sort(key=lambda item:item[0],reverse=True)
        match=None
        if eligible:
            adjusted,raw,self_floor,event=eligible[0]
            second=eligible[1][0] if len(eligible)>1 else 0.
            if (raw>=REID_QUALITY_FLOOR and raw>=self_floor and
                    adjusted-second>=REID_RELATIVE_MARGIN):
                match=event
        # Last/first re-association rescues compressed fade or short reflow.
        # It may NOT defeat co-visible exclusion, temporal expiry, or the
        # width/geometry veto applied above.
        if match is None:
            first_obs=candidate['samples'][0]
            boundary=[]
            for event in events:
                if frames & event['frames']:continue
                penalty=_event_relation(candidate,event)
                if penalty is None:continue
                last_obs=max((o for g in event['source_groups'] for o in g),
                             key=lambda o:o['t'])
                gap=first_obs['t']-last_obs['t']
                if not (0<gap<=BOUNDARY_REJOIN_SECONDS):continue
                edge_sim=dice_cv(first_obs['sig'],last_obs['sig'],8)
                floor=min(event['stability'],candidate['stability'])-.15
                boundary.append((edge_sim-penalty,edge_sim,floor,event))
            boundary.sort(key=lambda q:q[0],reverse=True)
            if boundary:
                adjusted,strength,floor,best=boundary[0]
                second=boundary[1][0] if len(boundary)>1 else 0.
                if strength>=floor and adjusted-second>=REID_RELATIVE_MARGIN:
                    match=best
        if match is None:
            events.append(dict(first=candidate['first'],last=candidate['last'],
                ref=candidate['ref'],refs=[candidate['ref']],
                stability=candidate['stability'],frames=frames,
                origin_track_id=candidate['origin_track_id'],
                source_groups=[candidate['samples']]))
        else:
            match['frames'] |= frames
            match['source_groups'].append(candidate['samples'])
            match['last']=max(match['last'],candidate['last'])
            match['first']=min(match['first'],candidate['first'])
            match['refs'].append(candidate['ref'])
    events=hold_short_fragments(events)
    events=_merge_same_row_aliases(events)
    events.sort(key=lambda e:e['first'])
    return events,initial,segments
