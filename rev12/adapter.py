#!/usr/bin/env python3
"""Single-command, verified NOTE14 candidate detector/owner QA adapter.
This emits a stable JSON/TSV/PNG integration interface; it NEVER touches
Windows production, JS, GitHub or publisher. Not a certified renderer patch.
"""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
import cv2
from datetime import datetime,timezone
from pathlib import Path
from fractions import Fraction
from tempfile import NamedTemporaryFile

BASE=Path(__file__).resolve().parent
STAGE=BASE
ENGINE=STAGE/'engine'
ENV={**os.environ,'OMP_THREAD_LIMIT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
LOW=.84024 # Recomputed 1st percentile of 168 source-replay native same-event PNG Dice values; not calibrated probability or independent GT.
sys.path.insert(0,str(ENGINE))
from hud_signature import signature,dice

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with NamedTemporaryFile(mode='w',dir=path.parent,prefix='._',suffix='.json',delete=False,encoding='utf-8') as f:
  tmp=Path(f.name);json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def frozen_signature():
 manifest_path=BASE/'REV12_SOURCE_SHA256.json'
 manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
 bad=[x for x,y in manifest.items() if not (BASE/x).is_file() or sha(BASE/x)!=y]
 if bad:raise RuntimeError('REV12_MANIFEST_MISMATCH '+','.join(bad[:4]))
 if len(manifest)<9:raise RuntimeError('REV12_MANIFEST_INCOMPLETE')
 return sha(manifest_path)

def parse(path,start=2.5):
 sources={};owner={};quality={};events=[]
 for l in path.read_text(encoding='utf-8').splitlines():
  t=l.split('\t')
  if t[0]=='SOURCE':sources[int(t[1])]={'time_s':float(t[3]),'png':t[8],'width':int(t[6]),'height':int(t[7])}
  elif t[0]=='EVENT':events.append((int(t[1]),float(t[2])+start,[int(v) for v in t[5].split(',')]))
  elif l.startswith('# OWNERSHIP '):
   q=l.split();owner[int(q[2])]=q[3]
  elif l.startswith('# ONSET_QUALITY '):
   q=l.split();quality[int(q[2])]=(q[3],float(q[4]))
 out=[]
 for eid,onset,ids in events:
  pics=[sources[i] for i in ids]
  if len(pics)==0 or any(not Path(p['png']).is_file() or Path(p['png']).stat().st_size==0 for p in pics):
   raise RuntimeError('EVENT_PNG_INVARIANT_FAILURE event='+str(eid))
  ow=owner.get(eid,'UNKNOWN');onset_status,onset_score=quality.get(eid,('UNSPECIFIED',0.0))
  if ow=='OTHER':raise RuntimeError('UNEXCLUDED_OTHER_EVENT')
  # Appearance must not be confused with the independent onset-refinement score.
  # Compare native victim-pixel glyphs across two *same-event* PNG observations.
  images=[cv2.imread(p['png']) for p in pics[:2]]
  glyphs=[signature(im) if im is not None else None for im in images]
  appearance=float(dice(glyphs[0],glyphs[1],8)) if len(glyphs)>1 and all(v is not None for v in glyphs) else 0.0
  out.append({'event_id':eid,'onset_seconds':round(onset,6),'onset_status':onset_status,'owner':ow,
   'onset_refinement_score':onset_score,'appearance_support':round(appearance,5),
   'confidence_index':round(appearance,5),'confidence_index_calibrated':False,'confidence_basis':'source_replay_168_own_native_pair_png_dice_p01_0.84024_not_probability_not_independent_GT',
   'evidence':'LOW_EVIDENCE' if appearance<LOW else 'STANDARD_EVIDENCE',
   'publish':True,'png_paths':[p['png'] for p in pics], 'png_count':len(pics),
   'source_row_width':pics[0]['width'],'source_row_height':pics[0]['height']})
 return out

def claim_process_lock(path):
 """A one-byte OS lock is automatically released even after a process crash."""
 fp=path.open('a+b')
 try:
  if os.name=='nt':
   import msvcrt
   fp.seek(0,2)
   if fp.tell()==0:fp.write(b'0');fp.flush()
   fp.seek(0)
   msvcrt.locking(fp.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl
   fcntl.flock(fp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  return fp
 except OSError:
  fp.close()
  return None

def release_process_lock(fp):
 try:
  if os.name=='nt':
   import msvcrt
   fp.seek(0);msvcrt.locking(fp.fileno(),msvcrt.LK_UNLCK,1)
  else:
   import fcntl
   fcntl.flock(fp.fileno(),fcntl.LOCK_UN)
 finally:fp.close()

def complete_cache(summary,key):
 try:
  data=json.loads(summary.read_text())
  if data.get('key')!=key or data.get('status')!='COMPLETE':return None
  images=[p for e in data.get('events',[]) for p in e.get('png_paths',[])]
  checks=data.get('png_sha256')
  if not isinstance(checks,dict) or len(checks)!=len(images):return None
  if any(not Path(p).is_file() or Path(p).stat().st_size==0 or sha(p)!=checks.get(p) for p in images):return None
  return {**data,'cached':True}
 except (OSError,ValueError,TypeError,KeyError):return None

def work(src,dest,owner,manifest_sha,timeout):
 src_sha=sha(src)
 key=hashlib.sha256(f'{src_sha}|{manifest_sha}|{owner}|NOTE14_CANDIDATE_ADAPTER_V1'.encode()).hexdigest()
 dest.mkdir(parents=True,exist_ok=True)
 summary=dest/'summary.json'
 cached=complete_cache(summary,key) if summary.exists() else None
 if cached is not None:return cached
 lock_fp=claim_process_lock(dest/'._LOCK_FILE')
 if lock_fp is None:return {'source':src.name,'status':'RETRY_LOCKED','failure':'CONCURRENT_JOB_LOCKED','render_policy':'RETRY_NO_PUBLICATION','events':[],'cached':False}
 # Another worker may have finished while we were acquiring the lock.
 cached=complete_cache(summary,key) if summary.exists() else None
 if cached is not None:
  release_process_lock(lock_fp)
  return cached
 data={'source':src.name,'source_sha256':src_sha,'candidate_manifest_sha256':manifest_sha,'key':key,
   'owner_global':owner,'status':'SAFE_NO_OVERLAY','publishable':False,'render_policy':'NO_OVERLAY','events':[],
   'cached':False,'started_utc':datetime.now(timezone.utc).isoformat()}
 try:
  probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-show_entries','format=duration','-of','json',str(src)],capture_output=True,text=True,timeout=20,check=True)
  met=json.loads(probe.stdout);duration=float(met['format']['duration']);fps=float(Fraction(met['streams'][0]['avg_frame_rate']))
  if not 3<duration<120 or not 0<fps<240:raise ValueError('INVALID_VIDEO_METADATA')
  data['duration_seconds']=duration;data['source_fps']=fps
  run=dest/'runs'/key[:20];run.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,str(ENGINE/'physical_row_tracker.py'),'--video',str(src),'--start','2.5','--end',str(duration),
       '--source-dir',str(run/'pngs'),'--out',str(run/'events.tsv'),'--owner-name',owner]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,env=ENV)
  (run/'tracker.log').write_text(p.stdout+'\n'+p.stderr)
  data['tracker_rc']=p.returncode
  if p.returncode:raise RuntimeError('TRACKER_RC_'+str(p.returncode)+': '+p.stderr[-180:])
  cmd=[sys.executable,str(STAGE/'owner_gate.py'),str(run/'events.tsv'),'--out',str(run/'owner_verified.tsv'),
       '--owner',owner,'--log',str(run/'owner_gate.json')]
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=90,env=ENV)
  (run/'owner.log').write_text(p.stdout+'\n'+p.stderr)
  if p.returncode:raise RuntimeError('OWNER_GATE_RC_'+str(p.returncode)+': '+p.stderr[-180:])
  events=parse(run/'owner_verified.tsv')
  data.update(events=events,png_sha256={p:sha(p) for e in events for p in e['png_paths']},event_count=len(events),png_count=sum(e['png_count'] for e in events),low_evidence_events=sum(e['evidence']=='LOW_EVIDENCE' for e in events),
              unknown_owners=sum(e['owner']=='UNKNOWN' for e in events), owner_gate_path=str(run/'owner_gate.json'),
              event_tsv=str(run/'owner_verified.tsv'),status='COMPLETE',publishable=True,render_policy='EMIT_EVENTS_TO_EXTERNAL_RENDERER_NOT_CONNECTED')
 except Exception as exc:
  data.update(failure=type(exc).__name__+': '+str(exc),status='SAFE_NO_OVERLAY',publishable=False,render_policy='NO_OVERLAY')
 finally:
  data['finished_utc']=datetime.now(timezone.utc).isoformat()
  atomic_json(summary,data)
  release_process_lock(lock_fp)
 return data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
 ap.add_argument('--owner',required=True);ap.add_argument('--timeout',type=int,default=350);ap.add_argument('--strict-qa',action='store_true');a=ap.parse_args()
 if len(a.owner.strip())<3:ap.error('--owner must be the single global attacker name (>=3 chars)')
 sig=frozen_signature();sources=sorted(a.input.glob('*.mp4')) if a.input.is_dir() else [a.input]
 if not sources:raise SystemExit('NO_MP4_FOUND')
 a.output.mkdir(parents=True,exist_ok=True);result=[]
 for src in sources:
  r=work(src,a.output/src.stem,a.owner.strip(),sig,a.timeout);result.append(r)
  print(json.dumps({'video':src.name,'status':r['status'],'count':len(r.get('events',[])),'cached':r['cached'],'failure':r.get('failure')},ensure_ascii=False),flush=True)
 atomic_json(a.output/'batch.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'videos':len(result),'complete':sum(x['status']=='COMPLETE' for x in result),'safe_no_overlay':sum(x['status']=='SAFE_NO_OVERLAY' for x in result),'retry_locked':sum(x['status']=='RETRY_LOCKED' for x in result),'results':[{'source':r['source'],'status':r['status'],'key':r.get('key')} for r in result]})
 if any(x['status']=='RETRY_LOCKED' for x in result):return 75
 return 2 if a.strict_qa and any(x['status']!='COMPLETE' for x in result) else 0
if __name__=='__main__':sys.exit(main())
