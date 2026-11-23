"""Independent 60 Hz four-edge confirmation of disconnected weak 10 Hz cohorts.

Never promotes weak 10 Hz detections directly. Only full detector rectangles
validated in TWO separated 60 Hz clusters with consistent native victim glyph
and independently stable geometry may establish a new physical event.
This is an ISOLATED experimental rev8 proposal, not production acceptance.
"""
import collections
import numpy as np
from hud_detector import source_frames,detect,RedCalibration
from hud_association import dice_cv
from hud_events import _strong_confirmed
from hud_signature import signature


def clusters(obs):
    out=[]
    for o in sorted(obs,key=lambda q:q['t']):
        if not out or o['t']-out[-1][-1]['t']>.16:out.append([o])
        else:out[-1].append(o)
    return [g for g in out if len({round(o['t']*10) for o in g})>=3
            and g[-1]['t']-g[0]['t']>=.18]


def independent_60_confirm(video, cluster, ref, ffmpeg):
    model=RedCalibration()
    med_y=float(np.median([o['y'] for o in cluster]))
    med_w=float(np.median([o['w'] for o in cluster]))
    med_right=float(np.median([o['x']+o['w'] for o in cluster]))
    med_h=float(np.median([o['h'] for o in cluster]))
    support=[]
    start=max(0,cluster[0]['t']-.10)
    end=cluster[-1]['t']+.20
    for t,img in source_frames(video,60,start=start,end=end,ffmpeg=ffmpeg):
        choices=[]
        for box in detect(img,model):
            if abs(box['y']-med_y)>med_h*.22:continue
            if abs(box['w']-med_w)>med_h*.20:continue
            if abs(box['x']+box['w']-med_right)>med_h*.22:continue
            sig=signature(box['roi'])
            score=dice_cv(sig,ref,5)
            if score>=.82:choices.append(score)
        if choices:support.append((t,max(choices)))
    # Each cohort needs independently full four-edge source geometry for
    # at least 0.18 s, with the glyph strongly tied to its native 10 Hz ref.
    if len(support)<10 or support[-1][0]-support[0][0]<.18:
        return False,len(support)
    coherent=any(support[i+9][0]-support[i][0]<=.24 for i in range(len(support)-9))
    return coherent and float(np.median([q for _,q in support]))>=.86,len(support)


def rescue_weak_events(video,events,segments,ffmpeg):
    used={id(o) for e in events for g in e['source_groups'] for o in g}
    for seg in sorted(segments,key=lambda x:x['first']):
        observations=seg['samples']
        if any(id(o) in used for o in observations):continue
        if len(observations)<6 or observations[-1]['t']-observations[0]['t']<.60:continue
        groups=clusters(observations)
        if len(groups)<2:continue
        # Two separated real observation cohorts, not duplicated 60fps images
        # or one fleeting four-edge false positive.
        groups=[g for g in groups if g[-1]['t']-g[0]['t']>=.18]
        if len(groups)<2:continue
        verified=[]
        for group in groups:
            yes,n=independent_60_confirm(video,group,seg['ref'],ffmpeg)
            if yes:verified.append((group,n))
            if len(verified)>=2:break
        if len(verified)<2:continue
        first,last=verified[0][0],verified[-1][0]
        if last[0]['t']-first[-1]['t']<.30:continue
        # Prevent an older same-row event or physical re-ID from being birthed
        # twice; distinct simultaneous rows are allowed.
        duplicate=False
        ref=seg['ref']
        own_med_y=float(np.median([o['y'] for o in observations]))
        own_right=float(np.median([o['x']+o['w'] for o in observations]))
        own_h=float(np.median([o['h'] for o in observations]))
        for e in events:
            seen=[o for g in e['source_groups'] for o in g]
            common=[o for o in seen if abs(o['t']-seg['first'])<.5 or abs(o['t']-seg['last'])<.5]
            if not common:continue
            ay=float(np.median([o['y'] for o in common]))
            ar=float(np.median([o['x']+o['w'] for o in common]))
            if abs(ay-own_med_y)<own_h*.2 and abs(ar-own_right)<own_h*.25:
                if max((dice_cv(ref,r,5) for r in e['refs']),default=0.)>=.73:
                    duplicate=True;break
        if duplicate:continue
        timestamps={round(o['t']*10) for o in observations}
        evt={'first':seg['first'],'last':seg['last'],'ref':ref,'refs':[ref],
             'stability':seg['stability'],'frames':timestamps,
             'origin_track_id':seg['origin_track_id'],
             'source_groups':[observations],
             'birth_provenance':'INDEPENDENT_TWO_COHORT_60FPS_FOUR_EDGE'}
        events.append(evt)
        used.update(id(o) for o in observations)
        print('INDEPENDENT_60FPS_FOUR_EDGE_CONFIRMED obs=%d cohorts=%d duration=%.2f' %
              (len(observations),len(verified),seg['last']-seg['first']),flush=True)
    events.sort(key=lambda e:e['first'])
    return events
