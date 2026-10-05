#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser(); ap.add_argument('--results-dir', required=True); a=ap.parse_args()
r=Path(a.results_dir)
files=sorted(p for p in r.rglob('*') if p.is_file() and p.name not in {'MANIFEST.sha256','manifest.json'})
rows=[{'path':str(p.relative_to(r)), 'bytes':p.stat().st_size, 'sha256':sha256(p)} for p in files]
(r/'manifest.json').write_text(json.dumps(rows,indent=2))
(r/'MANIFEST.sha256').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in rows))
print(f'wrote manifest for {len(rows)} files')
