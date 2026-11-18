"""Reproducible PathSnap candidate dataset; reference labels are derived proxies."""
from pathlib import Path
import argparse, hashlib, json, csv
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

FEATURES = ['distance_m','heading_cos','heading_available','segment_length_m',
            'projection_fraction','speed_mps','previous_gap_s','gps_step_m',
            'candidate_distance_rank','distance_minus_nearest_m']

def project_segments(points, starts, ends):
    delta = ends-starts
    fraction = np.clip(np.sum((points-starts)*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-12),0,1)
    distance = np.linalg.norm(points-(starts+fraction[...,None]*delta),axis=-1)
    return distance, fraction

def candidates(points, starts, ends, k=8):
    """Exact nearest segments: midpoint bound includes every segment within seed radius."""
    mids=(starts+ends)/2; half=np.linalg.norm(ends-starts,axis=1)/2
    tree=cKDTree(mids); count=min(k,len(starts))
    _, seeds=tree.query(points,k=count)
    seeds=np.asarray(seeds).reshape(len(points),count)
    result=[]; distances=[]; fractions=[]
    for p, seed in zip(points,seeds):
        ds,_=project_segments(p,starts[seed],ends[seed])
        ids=np.asarray(tree.query_ball_point(p,float(ds.max()+half.max()+1e-7)),dtype=int)
        d,f=project_segments(p,starts[ids],ends[ids]); order=np.lexsort((ids,d))[:count]
        result.append(ids[order]); distances.append(d[order]); fractions.append(f[order])
    return np.asarray(result),np.asarray(distances),np.asarray(fractions)

def align_reference(points, starts, ends, route):
    """Minimum sum squared distance with nondecreasing route occurrence indices.

    This is a label derivation heuristic, not verified observation-level truth.
    Free start/end and skipped occurrences support partially observed routes.
    """
    dist,_=project_segments(points[:,None,:],starts[route][None,:,:],ends[route][None,:,:])
    back=np.zeros(dist.shape,dtype=np.int32); costs=dist[0]**2
    for t in range(1,len(points)):
        # Prefix argmin, retaining earliest index on ties.
        prefix=np.minimum.accumulate(costs)
        indices=np.arange(len(route),dtype=np.int32)
        record=np.r_[True,costs[1:]<prefix[:-1]]
        arg=np.maximum.accumulate(np.where(record,indices,0))
        back[t]=arg; costs=dist[t]**2+prefix
    idx=np.empty(len(points),dtype=np.int32);idx[-1]=int(np.argmin(costs))
    for t in range(len(points)-1,0,-1): idx[t-1]=back[t,idx[t]]
    labels=route[idx]; residual=dist[np.arange(len(points)),idx]
    # Local geometric ambiguity among different reference arc IDs.
    other=np.where(route[None,:]!=labels[:,None],dist,np.inf).min(axis=1)
    margin=other-residual
    return labels,idx,residual,margin

def process(folder,out,fold):
    tid=folder.name
    files={ext:folder/f'{tid}.{ext}' for ext in ['track','nodes','arcs','route']}
    raw={ext:np.loadtxt(p,ndmin=2) for ext,p in files.items()}
    gps=raw['track']; nodes=raw['nodes']; arcs_raw=raw['arcs'];route_raw=raw['route'].ravel()
    assert gps.shape[1]==3 and nodes.shape[1]==2 and arcs_raw.shape[1]==2
    assert np.isfinite(nodes).all() and np.isfinite(arcs_raw).all() and np.isfinite(route_raw).all()
    assert np.equal(arcs_raw,np.floor(arcs_raw)).all() and np.equal(route_raw,np.floor(route_raw)).all()
    arcs=arcs_raw.astype(int);route=route_raw.astype(int)
    assert len(route)>0 and arcs.min()>=0 and arcs.max()<len(nodes) and route.min()>=0 and route.max()<len(arcs)
    valid=np.isfinite(gps).all(axis=1)&(np.abs(gps[:,0])<=180)&(np.abs(gps[:,1])<=90)
    original=np.flatnonzero(valid);gps=gps[valid]
    order=np.argsort(gps[:,2],kind='stable');gps=gps[order];original=original[order]
    # Keep same-time observations; undefined speeds/headings receive zero plus availability flag.
    lon,lat=np.median(nodes,axis=0)
    def project(coords):
        radians=np.deg2rad(coords-np.array([lon,lat]))
        return 6371008.8*radians*np.array([np.cos(np.deg2rad(lat)),1.])
    xy=project(gps[:,:2]);nxy=project(nodes)
    starts=nxy[arcs[:,0]];ends=nxy[arcs[:,1]];delta=ends-starts;length=np.linalg.norm(delta,axis=1)
    ids,distance,fraction=candidates(xy,starts,ends)
    labels,occ,residual,margin=align_reference(xy,starts,ends,route)
    dt=np.r_[0,np.diff(gps[:,2])];movement=np.vstack([np.zeros(2),np.diff(xy,axis=0)]);step=np.linalg.norm(movement,axis=1)
    usable=(dt>0)&(dt<=30)&(step>=2)
    speed=np.divide(step,dt,out=np.zeros_like(step),where=dt>0)
    cos=np.sum(movement[:,None,:]*delta[ids],axis=-1)/np.maximum(step[:,None]*length[ids],1e-12)
    cos=np.where(usable[:,None],cos,0)
    shape=ids.shape;repeat=lambda x:np.broadcast_to(x[:,None],shape)
    X=np.stack([distance,cos,repeat(usable.astype(float)),length[ids],fraction,repeat(speed),repeat(dt),repeat(step),np.broadcast_to(np.arange(shape[1]),shape),distance-distance[:,:1]],axis=-1).astype(np.float32)
    y=(ids==labels[:,None]).astype(np.int8)
    reliable=(residual<=50)&(margin>=3)
    train_mask=reliable & y.any(axis=1)
    destination=out/folder.parent.name;destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/f'{tid}.npz',X=X,y=y,candidate_arc_ids=ids,reference_arc_id=labels,reference_occurrence=occ,reference_distance_m=residual,reference_margin_m=margin,label_reliable=reliable,training_observation_mask=train_mask,original_row=original,gps=gps,xy_m=xy,fold=np.int16(fold),projection_origin_lonlat=np.array([lon,lat]))
    graph=coo_matrix((np.ones(len(arcs)),(arcs[:,0],arcs[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    components,_=connected_components(graph,directed=True,connection='weak')
    present=y.any(axis=1)
    return dict(split=folder.parent.name,trip_id=tid,fold=fold,raw_observations=len(raw['track']),observations=len(gps),removed_invalid=int((~valid).sum()),reordered_rows=int(np.sum(order!=np.arange(len(order)))),duplicate_timestamps=int(np.sum(dt[1:]==0)),gaps_over_30s=int(np.sum(dt>30)),max_gap_s=float(dt.max()),nodes=len(nodes),arcs=len(arcs),zero_length_arcs=int(np.sum(length==0)),weak_components=components,isolated_nodes=int(np.sum(np.bincount(arcs.ravel(),minlength=len(nodes))==0)),route_segments=len(route),route_connected_fraction=float(np.mean(arcs[route[:-1],1]==arcs[route[1:],0])) if len(route)>1 else 1.,candidate_misses=int((~present).sum()),reliable_labels=int(reliable.sum()),training_observations=int(train_mask.sum()),reference_distance_median=float(np.median(residual)),reference_distance_p95=float(np.percentile(residual,95)),projection_latitude=float(lat),lon_min=float(gps[:,0].min()),lon_max=float(gps[:,0].max()),lat_min=float(gps[:,1].min()),lat_max=float(gps[:,1].max()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path('Student folder'));p.add_argument('--out',type=Path,default=Path('processed'));args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    train=sorted((args.raw/'train').iterdir());rng=np.random.default_rng(212210209208); perm=rng.permutation(len(train));folds={train[int(i)].name:int(j%5) for j,i in enumerate(perm)}
    rows=[];manifest={}
    for split in ['train','student_test']:
        for folder in sorted((args.raw/split).iterdir()):
            if not folder.is_dir():continue
            row=process(folder,args.out,folds.get(folder.name,-1));rows.append(row)
            for f in sorted(folder.iterdir()):
                if f.is_file():manifest[str(f.relative_to(args.raw))]=hashlib.sha256(f.read_bytes()).hexdigest()
            print(split,folder.name,'observations',row['observations'],'usable',row['training_observations'],flush=True)
    with (args.out/'inventory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.out/'schema.json').write_text(json.dumps(dict(features=FEATURES,label_kind='derived_reference_proxy',k=8,alignment='monotone route distance minimisation',reliable_max_distance_m=50,reliable_min_margin_m=3,seed=212210209208,folds=5),indent=2))
    (args.out/'raw_sha256.json').write_text(json.dumps(manifest,indent=2))
    print('Finished',len(rows),'trips',flush=True)

if __name__=='__main__':main()
