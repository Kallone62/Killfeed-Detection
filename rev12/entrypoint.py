#!/usr/bin/env python3
"""Rev12 fail-closed, idempotent, per-video entrypoint (no V14 render changes).

All engine and owner-gate source bytes are verified on EVERY invocation.
SAFE_NO_OVERLAY is a render-policy signal, never proof of killfeed success.
RETRY_LOCKED is retryable and MUST NOT be published as processed.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
EXPECTED_MANIFEST_SHA = '2e3c7a1931c21833712156a44bbe4d9cff21b746b6ab0ee63c2632a1740087de'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify():
    source_manifest = ROOT / 'REV12_SOURCE_SHA256.json'
    if sha256(source_manifest) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError('REV12_MANIFEST_SHA256_MISMATCH')
    records = json.loads(source_manifest.read_text(encoding='utf-8'))
    if len(records) != 10:
        raise RuntimeError('REV12_SOURCE_MANIFEST_COUNT')
    for name, digest in records.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path) != digest:
            raise RuntimeError('REV12_SOURCE_SHA256_MISMATCH: ' + name)
    return EXPECTED_MANIFEST_SHA


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                            prefix='.rev12-', suffix='.tmp', delete=False) as file:
        temp = Path(file.name)
        try:
            json.dump(obj, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def safe(src, reason):
    return {'source':src.name,'source_path':str(src),'status':'SAFE_NO_OVERLAY',
            'publishable':False,'render_policy':'NO_OVERLAY','events':[],
            'event_count':0,'png_count':0,'cached':False,'failure':reason,
            'failure_utc':datetime.now(timezone.utc).isoformat()}


def process_one(src, output, owner, timeout, adapter, manifest_error):
    dest = output / src.stem
    try:
        if manifest_error:
            result = safe(src, manifest_error)
        else:
            result = adapter.work(src, dest, owner, EXPECTED_MANIFEST_SHA, timeout)
            if not isinstance(result, dict) or result.get('status') not in ('COMPLETE','SAFE_NO_OVERLAY','RETRY_LOCKED'):
                raise ValueError('INVALID_ADAPTER_RESULT')
            if result['status'] == 'SAFE_NO_OVERLAY':
                result.update(publishable=False, render_policy='NO_OVERLAY', events=[])
            if result['status'] == 'RETRY_LOCKED':
                result.update(publishable=False, render_policy='RETRY_NO_PUBLICATION')
    except BaseException as ex:
        result = safe(src, 'UNHANDLED_' + type(ex).__name__ + ': ' + str(ex)[:320])
        result['traceback'] = traceback.format_exc(limit=7)[-3200:]
    try:
        # A losing worker cannot clobber another worker's successful result.
        if result['status'] != 'RETRY_LOCKED':
            atomic_json(dest / 'rev12_final.json', result)
    except BaseException as ex:
        result = safe(src, 'RESULT_WRITE_' + type(ex).__name__ + ': ' + str(ex)[:200])
        print('REV12_WRITE_ERROR ' + repr(ex), file=sys.stderr)
    print(json.dumps({'source':src.name,'status':result['status'],
                      'events':len(result.get('events',[])),
                      'failure':result.get('failure')}, ensure_ascii=False), flush=True)
    return result


def run(sources, output, owner='Descrip', timeout=350):
    os.environ['OMP_THREAD_LIMIT'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        verify()
        sys.path.insert(0,str(ROOT / 'engine'))
        import adapter
        failure = None
    except BaseException as ex:
        adapter = None
        failure = 'MANIFEST_OR_IMPORT_' + type(ex).__name__ + ': ' + str(ex)[:300]
    rows = [process_one(Path(src), Path(output), owner, timeout, adapter, failure)
            for src in sources]
    batch = {'total':len(rows),'complete':sum(r['status']=='COMPLETE' for r in rows),
             'safe_no_overlay':sum(r['status']=='SAFE_NO_OVERLAY' for r in rows),
             'retry_locked':sum(r['status']=='RETRY_LOCKED' for r in rows),
             'all_complete':bool(rows) and all(r['status']=='COMPLETE' for r in rows),
             'source_manifest_sha256':EXPECTED_MANIFEST_SHA,
             'release_qa':'100/100_PASS_400_PARTIAL_163/400',
             'results':[{'source':r['source'],'status':r['status'],
                         'event_count':len(r.get('events',[])),
                         'failure':r.get('failure')} for r in rows]}
    try:
        atomic_json(Path(output) / 'rev12_batch.json', batch)
    except BaseException as ex:
        print('REV12_BATCH_LOG_ERROR ' + repr(ex), file=sys.stderr)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--owner', default=os.environ.get('DCS_OWNER','Descrip'))
    ap.add_argument('--timeout', type=int, default=350)
    ap.add_argument('--strict-qa', action='store_true')
    args = ap.parse_args()
    if not args.owner or len(args.owner.strip()) < 3:
        ap.error('Global --owner must be >=3 chars')
    sources = sorted(args.input.glob('*.mp4')) if args.input.is_dir() else [args.input]
    if not sources:
        print('NO_MP4_FOUND', file=sys.stderr)
        return 2
    rows = run(sources, args.output, args.owner.strip(), args.timeout)
    if rows['retry_locked']:
        return 75
    if args.strict_qa and not rows['all_complete']:
        return 2
    return 0

if __name__ == '__main__':
    try:
        code = main()
    except BaseException as ex:
        print('REV12_FATAL ' + repr(ex), file=sys.stderr)
        code = 2
    sys.exit(code)
