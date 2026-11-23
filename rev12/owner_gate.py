#!/usr/bin/env python3
"""Stable owner gate AFTER saved source PNGs, not while tracking moving rectangles.

Uses one GLOBAL configured attacker name for the input batch. Two independent
consistent complete non-owner OCR readings are required before an event is
excluded. Unknown owners are logged, never silently vetoed. No count/GT/title
inputs, no video-specific rules, and no modifications to V11 tracking or onsets.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parent/'engine'))
from hud_ownership import assess_ownership


def validate_and_rewrite(source_tsv:Path, output_tsv:Path, owner_name:str, log_json:Path):
    if not owner_name or len(owner_name.strip())<3:
        raise ValueError('GLOBAL_OWNER_NAME_REQUIRED')
    source_tsv=Path(source_tsv);output_tsv=Path(output_tsv);log_json=Path(log_json)
    lines=source_tsv.read_text(encoding='utf-8').splitlines()
    source={};groups=[];current=[]
    for line in lines:
        if line.startswith('SOURCE\t'):
            w=line.split('\t');source[int(w[1])]=w[8]
            current.append(line)
        elif line.startswith('STATE\t') or line.startswith('# ONSET_QUALITY ') or line.startswith('# OWNERSHIP '):
            current.append(line)
            if line.startswith('# OWNERSHIP '):
                groups.append(current);current=[]
        elif line.startswith('EVENT\t'):
            current.append(line)
        elif line.startswith('# MODE='):
            continue
        elif line.strip():
            raise ValueError('UNRECOGNIZED_REPORT_LINE '+line[:120])
    if current or not groups:raise ValueError('INCOMPLETE_EVENT_GROUPS')
    parsed=[];imgs=[]
    for group in groups:
        ev=[line.split('\t') for line in group if line.startswith('EVENT\t')]
        if len(ev)!=1 or len(ev[0])!=6:raise ValueError('EVENT_GROUP_MALFORMED')
        values=ev[0];eid=int(values[1]);ids=[int(v) for v in values[5].split(',')]
        batch=[]
        for sid in ids[:2]:
            image=cv2.imread(source[sid])
            if image is None:raise ValueError('SOURCE_PNG_MISSING '+source[sid])
            batch.append(image)
        parsed.append((eid,group,ids));imgs.append(batch)
    owners,explain=assess_ownership(imgs,owner_name=owner_name)
    rows=['# MODE=PHYSICAL_HUD_TRACKING_10FPS_HUNGARIAN_LOST_HYSTERESIS_OWNER_CONSENSUS_POSTPNG']
    decision=[];new_sid=0;new_event=0
    for i,(eid,group,ids) in enumerate(parsed):
        state=owners[i];reads=explain['reads'][i]
        decision.append({'raw_event':eid,'owner':state,'ocr_readings':reads,'original_source_ids':ids})
        if state=='OTHER':continue
        new_event+=1;mapping={}
        for line in group:
            parts=line.split('\t')
            if parts[0]=='SOURCE':
                new_sid+=1;mapping[int(parts[1])]=new_sid
                parts[1]=str(new_sid);rows.append('\t'.join(parts))
            elif parts[0]=='STATE':
                parts[1]=str(mapping[int(parts[1])]);parts[3]=str(new_event)
                rows.append('\t'.join(parts))
            elif parts[0]=='EVENT':
                parts[1]=str(new_event);parts[3]=str(mapping[int(parts[3])]);parts[5]=','.join(str(mapping[k]) for k in ids)
                rows.append('\t'.join(parts))
            elif line.startswith('# ONSET_QUALITY '):
                chunks=line.split();chunks[2]=str(new_event);rows.append(' '.join(chunks))
            elif line.startswith('# OWNERSHIP '):
                rows.append(f'# OWNERSHIP {new_event} {state}')
    summary={'owner_global':owner_name,'input_events':len(parsed),'output_events':new_event,
             'other_excluded':sum(x=='OTHER' for x in owners),'unknown_retained':sum(x=='UNKNOWN' for x in owners),
             'decisions':decision,'original_report_sha256':hashlib.sha256(source_tsv.read_bytes()).hexdigest(),
             'warning':'OCR ownership is not independent kill/onset ground truth'}
    output_tsv.parent.mkdir(exist_ok=True,parents=True);log_json.parent.mkdir(exist_ok=True,parents=True)
    output_tsv.write_text('\n'.join(rows)+'\n',encoding='utf-8');log_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source_tsv',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--owner',required=True);ap.add_argument('--log',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(validate_and_rewrite(a.source_tsv,a.out,a.owner,a.log),ensure_ascii=False))
if __name__=='__main__':main()
