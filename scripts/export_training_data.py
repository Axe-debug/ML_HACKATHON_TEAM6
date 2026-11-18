"""Create a public training handoff without coordinate or absolute-time metadata.

Original full local artifacts remain untouched. Output is a different export directory.
"""
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
KEYS={'X','y','candidate_arc_ids','reference_arc_id','reference_occurrence','reference_distance_m','reference_margin_m','label_reliable','training_observation_mask','fold'}
def export(source,destination):
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Use a separate export directory')
    destination.mkdir(parents=True,exist_ok=True)
    for split in ['train','student_test']:
        (destination/split).mkdir(exist_ok=True)
        for p in sorted((source/split).glob('*.npz')):
            with np.load(p,allow_pickle=False) as d:
                np.savez_compressed(destination/split/p.name,**{k:d[k] for k in sorted(KEYS)})
    with (source/'inventory.csv').open() as f:
        rows=list(csv.DictReader(f))
    columns=[k for k in rows[0] if k not in {'lon_min','lon_max','lat_min','lat_max','projection_latitude','epsg'}]
    with (destination/'inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r[k] for k in columns} for r in rows)
    schema=json.loads((source/'schema.json').read_text());schema['export_kind']='training_only_no_coordinates_or_absolute_times';schema['export_keys']=sorted(KEYS)
    (destination/'schema.json').write_text(json.dumps(schema,indent=2))
    shutil.copyfile(source/'raw_sha256.json',destination/'raw_sha256.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='processed');p.add_argument('--destination',required=True);a=p.parse_args();export(a.source,a.destination)
